# MolmoAct2 servers

The maintained adapter is `servers/serve_molmoact2.py`. It loads a local Hugging
Face snapshot and uses the checkpoint's `predict_action` implementation for
normalization and model preprocessing. The existing external `/act` servers
remain compatible with Karma's legacy client contract.

## Prepare the GPU environment

Use a separate [MolmoAct2 checkout](https://github.com/allenai/MolmoAct2) and its
model dependencies. The upstream `examples/yam/host_server_yam.py` was inspected
on 2026-11-08 as the API reference. There is no recorded immutable revision of the
previous GPU deployment in Karma; do not describe current upstream main as a
reproduction of that environment. Record the actual upstream commit, package
versions, checkpoint snapshot revision, GPU, and launch arguments in deployment
logs before promoting a new setup.

The source revision inspected for these instructions is
`6070080a20321b4f498ab30f28e1d09ac465edb7`. This is a setup reference, not a
claim that it is the prior physical deployment revision.

```bash
git clone https://github.com/allenai/MolmoAct2.git
cd MolmoAct2
git checkout 6070080a20321b4f498ab30f28e1d09ac465edb7
uv sync --python 3.12
# Choose and record an immutable model snapshot revision.
uv run hf download allenai/MolmoAct2-BimanualYAM \
  --revision YOUR_CHECKPOINT_COMMIT --local-dir outputs/models/MolmoAct2-BimanualYAM
# Save the resolved software environment with deployment artifacts.
uv pip freeze > deployment-packages.txt
```

The upstream project selects CUDA 12.8 PyTorch wheels, Torch 2.11.0,
Torchvision 0.26.0, and Transformers 4.57.x. Check driver compatibility on the
GPU host. Run the server launch below with `uv run --no-sync python` from this
environment. For SO101, change the download repository and local directory to
`allenai/MolmoAct2-SO100_101`.

Download an immutable snapshot of `allenai/MolmoAct2-BimanualYAM` or
`allenai/MolmoAct2-SO100_101`, including its configuration, processor, model code,
weights, and `norm_stats.json`. Install `json-numpy` and Pillow in that environment.
The adapter loads local checkpoint code with `trust_remote_code=True`.

## Bimanual YAM

```bash
# Run using the Python from the GPU model environment.
python /path/to/karma/servers/serve_molmoact2.py \
  --checkpoint /path/to/MolmoAct2-BimanualYAM \
  --rig yam_bimanual --norm-tag yam_dual_molmoact2 \
  --device cuda:0 --dtype bfloat16 --cuda-graph \
  --dataset-gripper-open 0 --port 8202
```

YAM uses three cameras in top, left wrist, right wrist order and 14 state/action
values, left arm first. Karma's wire gripper is 1=open. The launch flag explicitly
declares the dataset convention: with `--dataset-gripper-open 0`, the adapter
inverts grippers once on input and output. A checkpoint trained on 1=open data
must use `--dataset-gripper-open 1`. Do not apply this conversion elsewhere.

## SO101

```bash
python /path/to/karma/servers/serve_molmoact2.py \
  --checkpoint /path/to/MolmoAct2-SO100_101 \
  --rig so101 --norm-tag so100_so101_molmoact2 \
  --device cuda:0 --dtype bfloat16 --port 8203
```

This checkpoint uses two views (`top`, `side`) and its own six-dimensional
training frame. Configure `--policy-frame` using the calibration procedure in
[the existing inference guide](../inference.md#policy-frames). The server receives
that frame directly. Its `gripper_open=0` metadata identifies Karma's canonical
SO101 convention before the affine frame conversion; the training-frame gripper
range is defined by that frame and the checkpoint statistics.

```bash
uv run karma inference --rig so101 \
  --policy-config configs/policies/molmoact2-so101.json \
  --policy-frame calibration/so101-right.policy-frame.json \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --camera top=/dev/video5 --camera side=/dev/video7 \
  --instruction "pick up the object"
```

## Warmup and compatibility

Both servers must pass warmup before listening. Probe `GET /act` from the robot
machine, then use `karma policy doctor`. Server metadata includes the actual
checkpoint path, normalization tag, representation, and action cadence.

Some earlier checkpoint code needed bf16 fixes inside `predict_action`. Karma
casts floating processor inputs to model dtype but does not rewrite cached model
source. If warmup fails in checkpoint code with a bf16 conversion error, update
to a verified fixed checkpoint revision or use `--dtype float32` with sufficient
VRAM. Keep the failing traceback and exact versions; do not start a partially
initialized server.

The common server supports JPEG and raw RGB. CUDA graph capture stays serialized
with inference. `--num-steps` defaults to 10; camera framing and normalization must
match training. The [common command guide](README.md) covers all three commands
and Viser controls. Existing physical SO101 calibration limitations still apply.
