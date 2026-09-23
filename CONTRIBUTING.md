# How to contribute to Karma

Everyone is welcome to contribute, and we value every contribution. Code is
not the only way to help reporting bugs, improving docs, and answering
questions are all valuable.

## Ways to Contribute

- **Fixing issues:** Resolve bugs or improve existing code.
- **New features:** New rigs, workflows, or CLI commands.
- **Extend:** Support for new arms, sensors, or policy backends.
- **Documentation:** Improve setup guides, docstrings, and `docs/`.
- **Feedback:** File issues for bugs or desired features.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/karma.git
cd karma
git remote add upstream https://github.com/SRA-VJTI/karma.git
```

### 2. Environment Installation

Follow the [Install once](README.md#install-once) section of the README:
Ubuntu, Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/getting-started/installation/), then

```bash
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh
UV_HTTP_TIMEOUT=180 uv sync --locked
uv run karma --help
```

## Submitting Issues & Pull Requests

- **Issues:** Describe the rig/workflow you're using (`yam_bimanual`,
  `so101`, `so101_bimanual`, …), the command you ran, and the full error or
  log output.
- **Pull requests:** Branch off `master` (don't work directly on it), keep
  the change scoped, and describe what changed and why in the PR
  description.

Thank you for contributing to Karma!
