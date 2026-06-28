#!/usr/bin/env bash
# Download lc0 (GPU cuDNN + a CPU fallback) and the Maia human rating networks
# into the "Chess Engines" folder. Safe to re-run (skips nothing, just re-fetches).
#
#   bash setup/download_engines.sh
#
set -uo pipefail
CE="/c/Users/Connor/Documents/Chess Engines"
LC0_VER="v0.32.1"
mkdir -p "$CE/lc0" "$CE/lc0/cpu" "$CE/networks" "$CE/networks/maia"

dl() { echo "[dl] $(basename "$1")"; curl -L --fail -sS -o "$1" "$2" || { echo "  FAILED: $2"; return 1; }; }
unzip_py() { python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; }

# lc0 — NVIDIA cuDNN build (self-contained DLLs; matches the user's RTX GPU)
if dl "$CE/lc0_gpu.zip" "https://github.com/LeelaChessZero/lc0/releases/download/$LC0_VER/lc0-$LC0_VER-windows-gpu-nvidia-cudnn.zip"; then
  unzip_py "$CE/lc0_gpu.zip" "$CE/lc0" && rm -f "$CE/lc0_gpu.zip" && echo "[lc0] GPU build ready"
fi
# lc0 — CPU fallback (works anywhere, slower)
if dl "$CE/lc0_cpu.zip" "https://github.com/LeelaChessZero/lc0/releases/download/$LC0_VER/lc0-$LC0_VER-windows-cpu-dnnl.zip"; then
  unzip_py "$CE/lc0_cpu.zip" "$CE/lc0/cpu" && rm -f "$CE/lc0_cpu.zip" && echo "[lc0] CPU build ready"
fi

# Maia human rating networks (lc0 weight files; ~1.2 MB each)
for elo in 1100 1200 1300 1400 1500 1600 1700 1800 1900; do
  dl "$CE/networks/maia/maia-$elo.pb.gz" "https://raw.githubusercontent.com/CSSLab/maia-chess/master/maia_weights/maia-$elo.pb.gz"
done
# A stronger human net (also a reasonable casual Leela default)
dl "$CE/networks/maia/maia-2200.pb.gz" "https://github.com/CallOn84/LeelaNets/raw/refs/heads/main/Nets/Maia%202200/maia-2200.pb.gz" || true

echo "DONE-ENGINES"
echo "--- lc0 binaries ---"; ls "$CE/lc0"/*.exe "$CE/lc0/cpu"/*.exe 2>/dev/null
echo "--- maia nets ---"; ls "$CE/networks/maia" 2>/dev/null
