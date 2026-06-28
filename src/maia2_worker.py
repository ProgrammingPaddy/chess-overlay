"""Maia 2 inference worker — runs under the ISOLATED Python env (torch + maia2).

The app's interpreter (Python 3.14) has no PyTorch, so it CAN'T import maia2. It
launches THIS script with the maia2 env's python and talks to it over a
newline-delimited JSON pipe on stdin/stdout. The model loads ONCE and stays warm
(~5 ms/inference on GPU) — that warm process IS the low-latency pipe.

This file imports ONLY the standard library + maia2, so it is runnable standalone
under the isolated env (it must NOT import any app modules).

Protocol — one compact JSON object per line:
  ready (once, after load):  {"ready": true, "device": "...", "type": "..."}
  request:   {"id": N, "top_k": K, "queries": [{"fen", "elo_self", "elo_oppo"}, ...]}
  response:  {"id": N, "results": [{"win_prob": f, "moves": [[uci, prob], ...]}, ...]}
  error:     {"id": N, "error": "..."}   or   {"ready": false, "error": "..."}

Batching all of a turn's positions into ONE request keeps the round-trip count to
one even for the opponent look-ahead.
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="rapid")          # "rapid" | "blitz"
    ap.add_argument("--device", default="gpu")          # "gpu" | "cpu"
    ap.add_argument("--save-root", required=True)        # where the model weights live
    args = ap.parse_args()

    # maia2/gdown print load progress to STDOUT, which would corrupt the JSON
    # protocol. Redirect stdout -> stderr for the whole import + load, then restore
    # the real stdout for responses only.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from maia2 import inference, model
        maia_model = model.from_pretrained(type=args.type, device=args.device,
                                            save_root=args.save_root)
        prepared = inference.prepare()
        # Warm CUDA kernels on a dummy position so the first REAL query is fast
        # (the cold first inference is ~700 ms; warm is ~5 ms).
        inference.inference_each(
            maia_model, prepared,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1500, 1500)
    except Exception as exc:                              # pragma: no cover - env dependent
        sys.stdout = real_stdout
        sys.stdout.write(json.dumps({"ready": False, "error": repr(exc)}) + "\n")
        sys.stdout.flush()
        return 1
    sys.stdout = real_stdout

    def emit(obj) -> None:
        real_stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        real_stdout.flush()

    emit({"ready": True, "device": args.device, "type": args.type})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        if req.get("cmd") == "quit":
            break
        rid = req.get("id")
        top_k = int(req.get("top_k", 5))
        results = []
        for q in req.get("queries", []):
            # Each query is isolated: maia2 raises on a position with no legal moves
            # (checkmate/stalemate) or an illegal FEN, and one bad query must NOT take
            # down the whole batch (that would look like an engine crash). A failed
            # query just yields no prediction.
            try:
                move_probs, win_prob = inference.inference_each(
                    maia_model, prepared, q["fen"], int(q["elo_self"]), int(q["elo_oppo"]))
                moves = sorted(((k, float(v)) for k, v in move_probs.items()),
                               key=lambda kv: -kv[1])[:top_k]
                results.append({"win_prob": float(win_prob), "moves": moves})
            except Exception:
                results.append({"win_prob": None, "moves": []})
        emit({"id": rid, "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
