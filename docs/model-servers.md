# MolmoAct2 model servers

Karma never loads a model. A small HTTP server in the MolmoAct2 environment
holds the checkpoint on a GPU and answers `GET /act` (health) and `POST /act`
(inference); Karma is the client. The two servers used with this repository
live in the MolmoAct2 checkout (`host_server_yam.py` and
`examples/so100/host_server_so100.py`), not here. This page describes what each
one exposes so the Karma side can be configured, and what to check when writing
another one.

| | YAM server | SO100/SO101 server |
| --- | --- | --- |
| Checkpoint | MolmoAct2 bimanual YAM | `allenai/MolmoAct2-SO100_101` |
| Default port | 8202 | 8203 |
| `norm_tag` | `yam_dual_molmoact2` | `so100_so101_molmoact2` |
| `state_dim` | 14 | 6 |
| `num_cameras` / image keys | 3: `top_cam`, `left_cam`, `right_cam` | 2: `top_cam`, `side_cam` (order does not matter to the model) |
| State frame | Karma radians, gripper 1=open | LeRobot v1 degrees, gripper 0–100 (needs `--policy-frame`) |
| Karma rig | `yam_bimanual` | `so101` with `--camera ... top=` and `side=` |
| Actions per request | chunk of absolute targets | 30 (`action_horizon` 30, `n_action_steps` 30) |

## Health reply

```json
{"status": "ok", "norm_tag": "...", "state_dim": N, "num_cameras": K}
```

Karma refuses to energize if any field disagrees with the rig; a server that
serves a different checkpoint than the one you think is the failure this
catches. The SO100 server also reports `repo_id`, `device` and `dtype`.

## Request handling

- Images arrive as `json_numpy` arrays: JPEG bytes by default (decode them), or
  raw H×W×3 RGB with `--raw-frames`. The SO100 server requires every declared
  key and answers HTTP 400 naming a missing one (`missing required field:
  'side_cam'`).
- `state` is a float32 vector in the checkpoint's own frame. For YAM that is
  Karma's wire; the YAM adapter converts the gripper between the dataset
  convention (0=open) and its 1=open contract. For SO100 Karma has already
  applied the policy frame, so the server passes the state straight to the
  model — do not convert again.
- `normalization_tag` selects the entry in `norm_stats.json` (`q01_q99`
  normalization for SO100). `num_steps` (default 10) is the flow-matching
  solver steps; `enable_cuda_graph` requests CUDA-graph inference, which took
  the YAM expert from seconds to ~0.5 s per chunk.
- The SO100 server lowercases the instruction and strips trailing sentence
  punctuation to match training preprocessing.
- Reply `{"actions": <N × state_dim>, "dt_ms": ...}` in the same codec; Karma
  reports `dt_ms` as GPU time in its per-chunk log line.

## Running the SO100 server

From the MolmoAct2 checkout, with the checkpoint under
`outputs/models/MolmoAct2-SO100_101`:

```bash
bash scripts/serve_so100.sh
```

It loads the model in bf16 on `cuda:0`, warms up with dummy frames, then
listens on `0.0.0.0:8203`. Measured on an RTX 4090: ~150 ms GPU per 30-action
chunk, ~0.2 s round trip on a LAN with two 640×480 JPEG views. Check from the
robot machine before starting Karma:

```bash
curl http://GPU_HOST:8203/act
```

## Writing a server for your own checkpoint

Serve the checkpoint's real normalization key in `norm_tag`, its true
`state_dim` and the number of camera views it was trained with, read exactly the
image keys Karma sends for those roles, and accept state in the frame the
training data used. If that frame is Karma's radians (a checkpoint fine-tuned on
Karma recordings), no `--policy-frame` is needed. Karma's contract table is in
[inference](inference.md#contract-per-rig).
