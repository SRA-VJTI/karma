"""Fetch and cache the visual meshes the packaged URDFs reference.

The wheel ships each arm's URDF but not its ``assets/*.stl``: the URDFs are here
for the gravity-compensation model, which needs link inertias and joint origins
and never needs geometry. Rendering the real arm therefore needs the meshes from
the vendor's robot description.

This module downloads them once into ``~/openpi-data/meshes/<Model>/`` -- the
same ``~/openpi-data`` root the run logs use -- so every later run is offline.
The mesh list is read out of the packaged URDF itself, so a URDF that gains or
renames a link needs no change here.

Nothing in this module touches a bus, and nothing else in the package imports
it: fetching is always something you asked for.

    uv run openpi-control-viz --fetch-meshes --model Yam
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .config import resolve_model_assets
from .exceptions import ConfigurationError
from .log_paths import log_dir

# Read straight out of the URDF's <mesh filename="..."/> attributes.
_MESH_REF_RE = re.compile(r'filename\s*=\s*"([^"]+)"')

_FETCH_TIMEOUT_S = 60


@dataclass(frozen=True, slots=True)
class MeshSource:
    """Where one model's visual meshes come from, and under what licence."""

    # Raw base URL of the vendor's assets directory, without a trailing slash.
    base_url: str
    licence: str
    attribution: str
    # Pinned so a fetch is reproducible and cannot change under a cached dir.
    revision: str = ""
    # Full URLs for mesh names the vendor keeps somewhere other than base_url,
    # or publishes under a different name. Two names may share one URL.
    alternate_urls: dict[str, str] = field(default_factory=dict)

    def url_for(self, filename: str) -> str:
        """Where the vendor publishes ``filename``, honouring any override."""
        alternate = self.alternate_urls.get(filename)
        if alternate is not None:
            return alternate
        return f"{self.base_url}/{filename}"


_I2RT_REVISION = "7b6d5016f05ca63f9ef0185b7143e63f2c7a5708"
_I2RT_RAW = f"https://raw.githubusercontent.com/i2rt-robotics/i2rt/{_I2RT_REVISION}"

# The wrist. i2rt's yam.urdf points its last link at assets/link_6_visual.stl
# and assets/link_6_collision.stl, and the arm's own assets directory ships
# neither: that link is the crank gripper's body, and its geometry sits under
# the gripper model instead. Only the collision name is published there -- but
# in this asset set every *_visual.stl is byte-for-byte its *_collision.stl
# (base_link and links 1-5 all are), so that one file is the visual mesh too,
# not a stand-in for a missing one. Without this the wrist rendered bare.
_YAM_WRIST_STL = f"{_I2RT_RAW}/i2rt/robot_models/gripper/crank_4310/assets/link_6_collision.stl"

# Only models whose vendor publishes meshes matching the packaged URDF's mesh
# names appear here. The YAM URDF is i2rt's yam.urdf with its last link renamed
# to end_link, so the 14 mesh names line up one-to-one.
MESH_SOURCES: dict[str, MeshSource] = {
    "Yam": MeshSource(
        base_url=f"{_I2RT_RAW}/i2rt/robot_models/arm/yam/assets",
        revision=_I2RT_REVISION,
        licence="MIT",
        attribution="I2RT Robotics — https://github.com/i2rt-robotics/i2rt",
        alternate_urls={
            "link_6_visual.stl": _YAM_WRIST_STL,
            "link_6_collision.stl": _YAM_WRIST_STL,
        },
    ),
}


@dataclass(slots=True)
class FetchReport:
    """What one :func:`fetch_meshes` call did."""

    model: str
    directory: Path
    fetched: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    # Referenced by the URDF but not published anywhere this source knows to
    # look. Those links render without geometry; the rest still draw. Not an
    # error -- but worth checking against the vendor's layout before accepting,
    # since a mesh can simply have moved (see _YAM_WRIST_STL).
    unavailable: list[str] = field(default_factory=list)

    @property
    def available(self) -> int:
        return len(self.fetched) + len(self.already_present)

    def summary(self) -> str:
        lines = [f"{self.model}: {self.available} meshes in {self.directory}"]
        if self.fetched:
            lines.append(f"  fetched {len(self.fetched)}")
        if self.already_present:
            lines.append(f"  already cached {len(self.already_present)}")
        if self.unavailable:
            lines.append(
                f"  not published upstream ({len(self.unavailable)}): "
                + ", ".join(sorted(self.unavailable))
            )
        return "\n".join(lines)


def mesh_cache_root() -> Path:
    """The directory holding every model's cached meshes."""
    return log_dir().parent / "meshes"


def mesh_cache_dir(model: str) -> Path:
    """Where ``model``'s meshes are cached. Not created by this call."""
    return mesh_cache_root() / model


def cached_mesh_dir(model: str) -> Path | None:
    """The model's cache directory when it holds at least one mesh, else None."""
    directory = mesh_cache_dir(model)
    if directory.is_dir() and any(directory.glob("*.stl")):
        return directory
    return None


def urdf_mesh_names(model: str, *, urdf: Path | None = None) -> tuple[str, ...]:
    """The mesh filenames a model's packaged URDF references, deduplicated.

    Only the basename matters: the URDFs use ``package://assets/<name>`` or a
    bare relative path, and both resolve against one flat mesh directory.
    """
    assets = resolve_model_assets(model, urdf=urdf)
    if assets.urdf is None:
        raise ConfigurationError(f"model {model!r} ships no URDF, so it references no meshes")
    text = assets.urdf.read_text()
    names = {Path(ref).name for ref in _MESH_REF_RE.findall(text)}
    return tuple(sorted(name for name in names if name))


def fetch_meshes(
    model: str,
    *,
    urdf: Path | None = None,
    force: bool = False,
    destination: Path | None = None,
) -> FetchReport:
    """Download ``model``'s visual meshes into the local cache.

    Needs the network on first run for a given model; afterwards the cache is
    enough. Files already present are left alone unless ``force`` is set. A mesh
    the vendor does not publish is reported, not raised -- the arm still renders
    with the links that do have geometry.
    """
    source = MESH_SOURCES.get(model)
    if source is None:
        known = ", ".join(sorted(MESH_SOURCES)) or "none"
        raise ConfigurationError(
            f"no mesh source is known for model {model!r} (known: {known}); "
            "pass mesh_dir=<path> to point at meshes you already have"
        )

    names = urdf_mesh_names(model, urdf=urdf)
    directory = Path(destination) if destination is not None else mesh_cache_dir(model)
    directory.mkdir(parents=True, exist_ok=True)
    report = FetchReport(model=model, directory=directory)

    # Names a previous fetch proved absent upstream, each against the URL it
    # tried. Remembering them keeps a repeat fetch from needing the network just
    # to collect the same 404s; recording the URL means a name we have since
    # learned to look for elsewhere is retried rather than written off.
    known_absent = _read_absent(directory) if not force else {}

    # Two URDF names can name one upstream file, so fetch per URL, not per name.
    by_url: dict[str, list[str]] = {}
    for name in names:
        by_url.setdefault(source.url_for(name), []).append(name)

    for url, sharing in by_url.items():
        wanted = []
        for name in sharing:
            if (directory / name).exists() and not force:
                report.already_present.append(name)
            elif known_absent.get(name) == url:
                report.unavailable.append(name)
            else:
                wanted.append(name)
        if not wanted:
            continue
        listed = ", ".join(wanted)
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed https vendor URL
                url, timeout=_FETCH_TIMEOUT_S
            ) as response:
                payload = response.read()
        except urllib.error.HTTPError as err:
            if err.code == 404:
                report.unavailable.extend(wanted)
                continue
            raise ConfigurationError(
                f"fetching {listed} for {model} failed: HTTP {err.code} {err.reason}"
            ) from err
        except OSError as err:
            raise ConfigurationError(
                f"fetching {listed} for {model} failed: {err}. "
                "The first run needs network access; later runs use the cache."
            ) from err
        for name in wanted:
            # Write via a temporary name so an interrupted fetch cannot leave a
            # truncated STL that later runs would treat as cached.
            target = directory / name
            partial = target.with_suffix(target.suffix + ".partial")
            partial.write_bytes(payload)
            partial.replace(target)
            report.fetched.append(name)

    _write_attribution(directory, model, source)
    _write_absent(directory, {name: source.url_for(name) for name in report.unavailable})
    return report


_ABSENT_FILE = "unavailable-upstream.txt"


def _read_absent(directory: Path) -> dict[str, str]:
    """Mesh names a previous fetch missed, each with the URL it asked for.

    A line without a URL was written before this file recorded one, so it says
    nothing about where we would look now: dropping it costs one repeat 404 and
    buys a cache that heals itself when a mesh source learns a better address.
    """
    marker = directory / _ABSENT_FILE
    if not marker.is_file():
        return {}
    absent = {}
    for line in marker.read_text().splitlines():
        name, _, url = line.strip().partition("\t")
        if name and url:
            absent[name] = url
    return absent


def _write_absent(directory: Path, absent: dict[str, str]) -> None:
    marker = directory / _ABSENT_FILE
    if not absent:
        marker.unlink(missing_ok=True)
        return
    marker.write_text("".join(f"{name}\t{absent[name]}\n" for name in sorted(absent)))


def _write_attribution(directory: Path, model: str, source: MeshSource) -> None:
    """Record where the cached meshes came from, beside them."""
    note = directory / "SOURCE.txt"
    lines = [
        f"{model} visual meshes fetched by openpi_control.meshes.",
        f"Source: {source.base_url}",
    ]
    if source.revision:
        lines.append(f"Revision: {source.revision}")
    lines += [f"Licence: {source.licence}", f"Attribution: {source.attribution}", ""]
    note.write_text("\n".join(lines))
