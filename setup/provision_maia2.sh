#!/usr/bin/env bash
# Provision an ISOLATED Python 3.11 environment for Maia 2 (CSSLab/maia2).
#
# Why isolated: the app runs on Python 3.14, which has no PyTorch wheels, and
# maia2 pins torch==2.4.0 + chess==1.10.0 (older than the app's). So Maia 2 lives
# in its own venv and the app talks to it through a subprocess pipe. The venv goes
# in the "Chess Engines" folder so all engine deps live in one place. Safe to re-run.
#
#   bash setup/provision_maia2.sh
#
set -uo pipefail
CE="/c/Users/Connor/Documents/Chess Engines"
ENV="$CE/maia2-env"
PY="$ENV/Scripts/python.exe"

if [ ! -f "$PY" ]; then
  echo "[maia2] creating venv with Python 3.11..."
  py -3.11 -m venv "$ENV" || { echo "FAILED: could not create the venv (need Python 3.11)"; exit 1; }
fi

"$PY" -m pip install --upgrade pip wheel
echo "[maia2] installing torch 2.4.0 + CUDA 12.1 (~2.5 GB, slow)..."
"$PY" -m pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu121 || { echo "FAILED: torch"; exit 1; }
echo "[maia2] installing maia2..."
"$PY" -m pip install maia2 || { echo "FAILED: maia2"; exit 1; }
# maia2 0.9 ships an EMPTY dependency list, so install its real runtime deps by
# hand. numpy MUST be <2 (torch 2.4.0 is built against numpy 1.x); pyyaml is also
# undeclared. Keep these in the isolated env only.
echo "[maia2] installing maia2 runtime deps..."
"$PY" -m pip install "numpy<2" pyyaml "chess==1.10.0" pandas einops gdown pyzstd requests tqdm \
  || { echo "FAILED: maia2 deps"; exit 1; }

echo "[maia2] verifying + pre-caching the model..."
SAVE_WIN="$(cygpath -w "$CE/maia2_models" 2>/dev/null || echo "$CE/maia2_models")"
MAIA2_SAVE="$SAVE_WIN" "$PY" - <<'PYEOF'
import os, chess
from maia2 import model, inference
m = model.from_pretrained(type="rapid", device="gpu", save_root=os.environ["MAIA2_SAVE"])
prepared = inference.prepare()
mp, wp = inference.inference_each(m, prepared, chess.Board().fen(), 1500, 1500)
print("maia2 OK — top:", max(mp, key=mp.get), "win_prob:", round(float(wp), 3))
PYEOF
echo "DONE-MAIA2"
