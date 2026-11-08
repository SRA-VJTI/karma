"""Isolated SDK worker. Stdout is JSONL RPC; no hardware imports or access."""

import contextlib
import json
import logging
import sys
import threading


def main():
    channel = sys.stdout
    write_lock = threading.Lock()
    infer_lock = threading.Lock()
    generation = -1
    policy_config = None

    def reply(request, result=None, error=None):
        message = {"id": request["id"], "result": result, "error": error}
        with write_lock:
            channel.write(json.dumps(message, allow_nan=False, default=lambda v: v.tolist()) + "\n")
            channel.flush()

    with contextlib.redirect_stdout(sys.stderr):
        import dream

        client = dream.DreamClient()

        def infer(request, captured=None):
            try:
                with infer_lock:
                    if request["generation"] != generation:
                        raise ValueError("obsolete inference generation")
                    # Anchor observation and commands together immediately before get_action.
                    result = client.get_action(commanded=request["commanded"])
                    horizon = len(result["q_l_chunk"])
                    if not 1 <= horizon <= 256:
                        raise ValueError("invalid action horizon")
                    reply(
                        request,
                        {
                            "left": result["q_l_chunk"],
                            "right": result["q_r_chunk"],
                            "timing": client.last_timing,
                            "mode": client.mode,
                            "refill": [
                                [client.refill_due(i, j) for j in range(horizon + 1)]
                                for i in range(horizon + 1)
                            ],
                        },
                    )
            except Exception as exc:
                if captured is not None:
                    captured.set()
                logging.error("Dream inference failed: %s", type(exc).__name__)
                reply(request, error=type(exc).__name__)

        try:
            for line in sys.stdin:
                request = json.loads(line)
                try:
                    op = request["op"]
                    if op == "policy":
                        client.policy(request["config"])
                        policy_config = request["config"]
                        reply(request, True)
                    elif op == "health":
                        models = client.get_models()
                        model = next(
                            (
                                m
                                for m in models
                                if (m.get("id") if isinstance(m, dict) else getattr(m, "id", None))
                                == request["checkpoint"]
                            ),
                            None,
                        )
                        if model is None:
                            raise ValueError("checkpoint not available to this account")

                        def get(k, default=None, model=model):
                            return (
                                model.get(k, default)
                                if isinstance(model, dict)
                                else getattr(model, k, default)
                            )

                        if not get("servable", True):
                            raise ValueError("checkpoint is not servable")
                        roles = get("cameras")
                        if roles is not None and set(roles) != {"overhead", "wrist_l", "wrist_r"}:
                            raise ValueError("checkpoint camera roles do not match YAM")
                        reply(request, {"status": "ok", "checkpoint": request["checkpoint"]})
                    elif op == "reset":
                        with infer_lock:
                            client.reset_episode()
                            if request["config"] != policy_config:
                                client.policy(request["config"])
                                policy_config = request["config"]
                            generation = request["generation"]
                        reply(request, True)
                    elif op == "tick":
                        if request["generation"] == generation:
                            tick = request["tick"]
                            client.set_observation(
                                tick["frames"], tick["state"], timestamp=tick["timestamp"]
                            )
                    elif op == "infer":
                        tick = request["tick"]
                        client.set_observation(
                            tick["frames"], tick["state"], timestamp=tick["timestamp"]
                        )
                        captured = threading.Event()
                        window = client._window
                        original = window.build_observation

                        def snapshot(original=original, captured=captured):
                            observation = original()
                            captured.set()
                            return observation

                        window.build_observation = snapshot
                        threading.Thread(
                            target=infer, args=(request, captured), daemon=True
                        ).start()
                        if not captured.wait(2):
                            raise TimeoutError("SDK failed to capture observation")
                        window.build_observation = original
                    elif op == "close":
                        break
                    else:
                        raise ValueError("unknown operation")
                except Exception as exc:
                    logging.error(
                        "Dream operation %s failed: %s", request.get("op"), type(exc).__name__
                    )
                    reply(request, error=type(exc).__name__)
        finally:
            client.close()


if __name__ == "__main__":
    main()
