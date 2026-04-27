# my-research

手術・看護手技動画を対象とした Video Temporal Grounding の研究コード。

## ディレクトリ構成

このリポジトリはコードのみを管理します。データ・結果は Google Drive で管理します。

```
research/               ← --base_dir に渡すルート
├── datasets/           ← 動画クリップ
├── experiments/
│   └── timelens/
│       ├── plans/      ← テストデータ JSON
│       └── results/    ← 推論結果
└── models/
    └── TimeLens-8B/    ← モデルウェイト
```

## セットアップ（初回のみ）

```bash
bash setup_timelens.sh
```

## 推論コマンド

### 自宅 PC（RTX 3060 12GB）

```fish
~/venvs/timelens/bin/python3 ~/ghq/github.com/kidotch/my-research/run_inference.py \
  --base_dir ~/univ/research \
  --test_data experiments/timelens/plans/test_data_60s_20260420.json
```

- `--fps` / `--total_pixels` 指定不要（デフォルト値で動作）
- 約50秒/サンプル

### 研究室 PC（RTX 2070 8GB, WSL2）

```fish
source ~/venvs/timelens/bin/activate.fish && python3 ~/ghq/github.com/kidotch/my-research/run_inference.py \
  --base_dir /mnt/d/kido/univ/research \
  --test_data experiments/timelens/plans/test_data_60s_20260420.json \
  --max_gpu_memory 6 \
  --fps 1 \
  --total_pixels 3145728
```

- Flash Attention 非対応のため `--fps 1 --total_pixels 3145728` でメモリ削減が必要
- 約180秒/サンプル
- 実行前に Whisper 等の残留プロセスが VRAM を食っていないか確認（`nvidia-smi`）

## オプション一覧

```
--base_dir        研究ルートディレクトリ（必須）
--test_data       テストデータ JSON（base_dir からの相対パスまたは絶対パス）
--model           モデルディレクトリ（省略時: models/TimeLens-8B）
--results_dir     結果保存先（省略時: experiments/timelens/results）
--fps             動画サンプリング FPS（省略時: 2）
--total_pixels    全フレームの合計ピクセル予算（省略時: 14680064）
                  8GB VRAM 向けには 3145728 を推奨
--max_gpu_memory  GPU に乗せるモデルの上限 GB（省略時: 5）
--quantize        量子化モード: none / int8 / int4（省略時: none）
```
