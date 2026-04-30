# vtg-scripts

手術・看護手技動画を対象とした Video Temporal Grounding の個人研究スクリプト集。

コードのみを管理します。データ・結果は Google Drive で管理します。

## ディレクトリ構成

```
vtg-scripts/
├── timelens/
│   ├── setup.sh          ← 環境構築
│   └── run_inference.py  ← 推論スクリプト
└── unitime/
    ├── setup.sh          ← 環境構築
    └── run_inference.py  ← 推論スクリプト
```

### Google Drive 側（--base_dir に渡すルート）

```
research/
├── datasets/           ← 動画クリップ
├── experiments/
│   ├── timelens/
│   │   ├── plans/      ← テストデータ JSON
│   │   └── results/    ← 推論結果
│   └── unitime/
│       ├── plans/      ← テストデータ JSON
│       └── results/    ← 推論結果
└── models/
    ├── TimeLens-8B/
    └── UniTime/
```

---

## TimeLens

### セットアップ

```bash
bash timelens/setup.sh
```

### 推論コマンド

**自宅 PC（RTX 3060 12GB）**

```fish
~/venvs/timelens/bin/python3 ~/ghq/github.com/kidotch/vtg-scripts/timelens/run_inference.py \
  --base_dir ~/univ/research \
  --test_data experiments/timelens/plans/test_data_60s_20260420.json
```

**研究室 PC（RTX 2070 8GB, WSL2）**

```fish
~/venvs/timelens/bin/python3 ~/ghq/github.com/kidotch/vtg-scripts/timelens/run_inference.py \
  --base_dir /mnt/d/kido/univ/research \
  --test_data experiments/timelens/plans/test_data_60s_20260420.json \
  --max_gpu_memory 6 \
  --fps 1 \
  --total_pixels 3145728
```

### オプション

```
--base_dir        研究ルートディレクトリ（必須）
--test_data       テストデータ JSON（base_dir からの相対パスまたは絶対パス）
--model           モデルディレクトリ（省略時: models/TimeLens-8B）
--results_dir     結果保存先（省略時: experiments/timelens/results）
--fps             動画サンプリング FPS（省略時: 2）
--total_pixels    全フレームの合計ピクセル予算（省略時: 14680064）
--max_gpu_memory  GPU に乗せるモデルの上限 GB（省略時: 5）
--quantize        量子化モード: none / int8 / int4（省略時: none）
```

---

## UniTime

### セットアップ前提

- UniTimeリポジトリ: `/workspace/UniTime`（または任意のパス）
- ベースモデル: `models/Qwen2-VL-7B-Instruct/`
- UniTimeウェイト: `models/UniTime/`（HuggingFace: zeqianli/UniTime）

```bash
git clone https://github.com/Lzq5/UniTime /workspace/UniTime
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir models/Qwen2-VL-7B-Instruct
huggingface-cli download zeqianli/UniTime --local-dir models/UniTime
```

### 推論コマンド

**Vast.ai**

```bash
python /workspace/vtg-scripts/unitime/run_inference.py \
  --base_dir /workspace/research \
  --test_data experiments/timelens/plans/test_data_60s_20260420.json \
  --unitime_repo /workspace/UniTime
```

### オプション

```
--base_dir        研究ルートディレクトリ（必須）
--test_data       テストデータ JSON（base_dir からの相対パスまたは絶対パス）
--unitime_repo    UniTimeリポジトリのパス（省略時: /workspace/UniTime）
--base_model      Qwen2-VL-7B-Instruct（省略時: models/Qwen2-VL-7B-Instruct）
--model           UniTimeウェイト（省略時: models/UniTime）
--results_dir     結果保存先（省略時: experiments/unitime/results）
--fps             動画サンプリング FPS（省略時: 2）
--clip_length     クリップ長フレーム数（省略時: 32）
--nf_short        この秒数以下はmrモード（省略時: 128）
```
