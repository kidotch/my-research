#!/bin/bash
# TimeLens セットアップスクリプト
# 実行方法: bash setup_timelens.sh [--timelens_dir /path/to/TimeLens]
# オプション:
#   --timelens_dir  TimeLensリポジトリのクローン先（デフォルト: スクリプトと同じ親ディレクトリ/TimeLens）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/../TimeLens"
ENV_NAME="timelens"

# 引数処理
while [[ $# -gt 0 ]]; do
    case $1 in
        --timelens_dir) REPO_DIR="$2"; shift 2 ;;
        *) echo "不明なオプション: $1"; exit 1 ;;
    esac
done

REPO_DIR="$(realpath -m "$REPO_DIR")"

echo "=== TimeLens セットアップ開始 ==="

# 1. リポジトリのクローン
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/4] リポジトリをクローン中..."
    git clone https://github.com/TencentARC/TimeLens.git "$REPO_DIR"
else
    echo "[1/4] リポジトリ既存 ($REPO_DIR) - スキップ"
fi

cd "$REPO_DIR"

# 2. conda環境の作成
echo "[2/4] conda環境を作成中 ($ENV_NAME)..."
conda create -n "$ENV_NAME" python=3.11 -y || echo "環境既存 - スキップ"

# 3. 依存パッケージのインストール
echo "[3/4] 依存パッケージをインストール中..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Blackwell GPU (sm_120) 対応のため cu128 を使用
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# 4. 動作確認
echo "[4/4] 動作確認..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA利用可能: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "=== セットアップ完了 ==="
echo "推論実行例:"
echo "  conda activate $ENV_NAME"
echo "  python run_inference.py --base_dir /path/to/research"
