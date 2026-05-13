#!/bin/bash
# TimeLens 推論環境セットアップ
# 使い方: bash setup_timelens.sh
#
# 動作確認済み環境:
#   Ubuntu 24.04 / Python 3.12 / CUDA 12.4
#   RTX 3060 (VRAM 12GB) / RTX 2070 (VRAM 8GB)

set -e

VENV_DIR="$HOME/venvs/timelens"
CUDA_INDEX="https://download.pytorch.org/whl/cu124"

echo "=== TimeLens 推論環境セットアップ ==="
echo "venv: $VENV_DIR"
echo ""

# 1. venv 作成
if [ -d "$VENV_DIR" ]; then
    echo "[1/3] venv 既存 ($VENV_DIR) — スキップ"
else
    echo "[1/3] venv 作成中..."
    python3 -m venv "$VENV_DIR"
    echo "      完了"
fi

PIP="$VENV_DIR/bin/pip"

# 2. パッケージインストール
echo "[2/3] パッケージインストール中..."
"$PIP" install --upgrade pip -q

echo "      PyTorch (cu124)..."
"$PIP" install torch torchvision --index-url "$CUDA_INDEX" -q

echo "      transformers / accelerate / bitsandbytes..."
"$PIP" install transformers accelerate bitsandbytes -q

echo "      qwen-vl-utils / av / decord / psutil..."
"$PIP" install qwen-vl-utils av decord psutil -q

echo "      完了"

# 3. 動作確認
echo "[3/3] 動作確認..."
"$VENV_DIR/bin/python" - <<'EOF'
import torch
print(f"  PyTorch:  {torch.__version__}")
print(f"  CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024**3
    print(f"  GPU:      {props.name}  ({vram:.1f} GB VRAM)")
    if vram <= 9:
        print(f"  推奨設定: --max_gpu_memory 4  (VRAM {vram:.0f}GB)")
    else:
        print(f"  推奨設定: --max_gpu_memory 5  (VRAM {vram:.0f}GB)")

import transformers
print(f"  transformers: {transformers.__version__}")
import bitsandbytes
print(f"  bitsandbytes: {bitsandbytes.__version__}")
import qwen_vl_utils
print(f"  qwen_vl_utils: ok")
import av
print(f"  av: ok")
EOF

echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "実行例 (VRAM 12GB):"
echo "  $VENV_DIR/bin/python run_inference.py \\"
echo "    --base_dir ~/univ/research \\"
echo "    --test_data experiments/timelens/plans/0514/senshi_hand-sani_0513.json"
echo ""
echo "実行例 (VRAM 8GB):"
echo "  $VENV_DIR/bin/python run_inference.py \\"
echo "    --base_dir ~/univ/research \\"
echo "    --test_data experiments/timelens/plans/0514/senshi_hand-sani_0513.json \\"
echo "    --max_gpu_memory 4"
