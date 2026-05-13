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
│   │   └── results/    ← 推論結果（日付サブフォルダ）
│   └── unitime/
│       ├── plans/      ← テストデータ JSON
│       └── results/    ← 推論結果（日付サブフォルダ）
└── models/
    ├── TimeLens-8B/
    ├── Qwen2-VL-7B-Instruct/
    └── UniTime/
```

---

## 環境要件と互換性

### ライブラリバージョン競合（重要）

TimeLens と UniTime は **transformers のバージョンが競合** するため、同一環境での同時実行不可。

| モデル | transformers | PyTorch | flash_attn |
|---|---|---|---|
| UniTime | ==4.49.0 | >=2.1（cu121推奨） | 必須（OOM回避） |
| TimeLens | >=4.50（Qwen3VL必要） | >=2.4 | 不要 |

### Vast.ai での運用方針

- **UniTime**: UniTime公式Dockerイメージを使用し、PyTorchはそのまま（cu121）
- **TimeLens**: UniTimeのDockerイメージ上でtransformersとPyTorchをアップグレード
- **順序**: UniTimeを先に実行 → transformers/PyTorchをアップグレード → TimeLens実行

### GPU別 PyTorch バージョン

| GPU | アーキテクチャ | 推奨 PyTorch |
|---|---|---|
| RTX 4090 / A100 など | sm_86〜sm_90 | cu121 または cu124 |
| RTX PRO 6000 / RTX 5090 など | Blackwell (sm_120) | cu128 必須 |

> **注意**: cu128 への不要なアップグレードは UniTime の flash_attn を破壊する。必ずGPUを確認してから決定する。

---

## TimeLens

### セットアップ

```bash
bash timelens/setup.sh
```

### 推論コマンド

**Vast.ai（UniTime後に実行する場合）**

```bash
# transformers と PyTorch をアップグレード（UniTime終了後）
pip install --upgrade transformers torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 -q

python /workspace/vtg-scripts/timelens/run_inference.py \
  --base_dir /workspace/research \
  --test_data experiments/timelens/plans/0514/senshi_hand-sani_0513.json
```

**自宅 PC（RTX 3060 12GB）**

```fish
~/venvs/timelens/bin/python3 ~/ghq/github.com/kidotch/vtg-scripts/timelens/run_inference.py \
  --base_dir ~/univ/research \
  --test_data experiments/timelens/plans/0514/senshi_hand-sani_0513.json
```

**研究室 PC（RTX 2070 8GB, WSL2）**

```fish
~/venvs/timelens/bin/python3 ~/ghq/github.com/kidotch/vtg-scripts/timelens/run_inference.py \
  --base_dir /mnt/d/kido/univ/research \
  --test_data experiments/timelens/plans/0514/senshi_hand-sani_0513.json \
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
```

モデルウェイトは Google Drive の `research/models/` に保存済み。

### 推論前の環境確認（Vast.ai）

```bash
# transformers を UniTime 用にダウングレード（TimeLens後に実行する場合）
pip install transformers==4.49.0 -q

# flash_attn の動作確認（必須: ないと長時間動画でOOM）
python -c "import flash_attn; print(flash_attn.__version__)"
```

### 推論コマンド

**Vast.ai**

```bash
python /workspace/vtg-scripts/unitime/run_inference.py \
  --base_dir /workspace/research \
  --test_data experiments/timelens/plans/0514/senshi_hand-sani_0513.json \
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
--nf_short        この秒数以下はmrモード（省略時: 128、-1は使用不可）
```

> **注意**: `--nf_short -1` は UniTime の inference.py のバグにより UnboundLocalError が発生する。使用しないこと。

---

## 既知の問題

| 問題 | 原因 | 対処 |
|---|---|---|
| UniTimeでOOM（長時間動画） | flash_attn未インストール時にSDPA使用でO(n²)メモリ | flash_attnを正しいPyTorchバージョン向けに再インストール |
| TimeLensで `qwen3_vl` 認識エラー | transformers 4.49.0ではQwen3VL非対応 | `pip install --upgrade transformers` |
| `--nf_short -1` でUnboundLocalError | UniTime inference.pyのバグ | 使用しない |
| rcloneが入っていない | UniTime Dockerイメージに含まれない | `apt install unzip -y && curl -s https://rclone.org/install.sh \| bash` |
