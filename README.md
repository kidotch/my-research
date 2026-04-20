# jichi-research

手術・看護手技動画を対象とした Video Temporal Grounding の研究コード。

## ディレクトリ構成

このリポジトリはコードのみを管理します。データ・結果は Google Drive で管理します。

```
research/               ← --base_dir に渡すルート（Google Drive からコピー）
├── datasets/           ← 動画クリップ
├── experiments/
│   └── timelens/
│       ├── plans/      ← テストデータ JSON（Google Drive 管理）
│       └── results/    ← 推論結果（Google Drive 管理）
└── models/
    └── TimeLens-8B/    ← モデルウェイト
```

## セットアップ

```bash
# 1. TimeLens 環境構築（初回のみ）
bash setup_timelens.sh

# 2. 推論実行
conda activate timelens
python run_inference.py --base_dir /path/to/research
```

## 動作確認済み環境

- GPU: NVIDIA RTX PRO 6000 (Blackwell, sm_120)
- PyTorch: 2.11.0+cu128
- Python: 3.11

## オプション

```
python run_inference.py \
  --base_dir     /path/to/research   # 必須: research ルートディレクトリ
  --test_data    path/to/test.json   # 省略時: experiments/timelens/plans/test_data_20260419.json
  --model        path/to/model       # 省略時: models/TimeLens-8B
  --results_dir  path/to/results     # 省略時: experiments/timelens/results
  --fps          2                   # 動画サンプリング FPS（省略時: 2）
```
