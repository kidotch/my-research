# TimeLens セットアップ計画

## 概要

**TimeLens** は動画中の特定行動がどの時間区間で発生しているかを検出する Video Temporal Grounding モデル。
自然言語クエリ（例：「手を消毒している」）と動画ファイルを入力し、`[[開始秒, 終了秒]]` を出力する。

- **リポジトリ**: https://github.com/TencentARC/TimeLens
- **論文**: CVPR 2026 / arXiv:2512.14698
- **開発元**: TencentARC

---

## ディレクトリ構成

```
/workspace/
├── setup_plans/
│   ├── timelens_setup.md   # このファイル
│   └── setup_timelens.sh   # セットアップスクリプト
├── TimeLens/               # クローン済みリポジトリ
└── research/               # データセット（Google Driveからsync済み）
    └── datasets/jichi/
        ├── annotations/    # JSONアノテーション
        ├── videos/         # MP4動画ファイル
        └── whisper/        # 文字起こしデータ
```

---

## 環境情報

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell WS |
| VRAM | 97,887 MiB（約96GB） |
| Python | 3.12 |
| conda | miniforge3 |
| インスタンスID | 35286783 |

---

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/TencentARC/TimeLens.git /workspace/TimeLens
cd /workspace/TimeLens
```

### 2. conda環境の作成

```bash
conda create -n timelens python=3.11 -y
conda activate timelens
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124
# Blackwell GPU (sm_120) 対応: cu128版に上書きアップグレード
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

### 4. モデルウェイトのダウンロード

```bash
pip install hf_transfer -q
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TencentARC/TimeLens-8B \
  --local-dir /workspace/models/TimeLens-8B \
  --local-dir-use-symlinks False
```

> モデルサイズ: 約17GB。ダウンロード先: `/workspace/models/TimeLens-8B/`

### 5. 動作確認

```bash
conda activate timelens
cd /workspace/TimeLens
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 推論の実行方法

### 入力フォーマット

- **動画**: MP4ファイル（例: `/workspace/research/datasets/jichi/videos/jichi2507/irou_001.mp4`）
- **クエリ**: 日本語または英語のテキスト（例: `"The person is sanitizing their hands"`）

### 出力フォーマット

```json
{
  "timestamps": [[12.5, 18.3]],
  "answers": "The event happens in 12.5 - 18.3 seconds."
}
```

### サンプルクエリ（本研究向け）

| 行動 | クエリ例（英語） |
|---|---|
| 手洗い | `"A person is washing their hands"` |
| 手指消毒 | `"A person is sanitizing their hands with alcohol"` |
| 手袋着用 | `"A person is putting on gloves"` |

---

## 環境の再現手順

新しいインスタンスで環境を再現する場合は `setup_timelens.sh` を実行：

```bash
bash /workspace/setup_plans/setup_timelens.sh
```

---

## 注意事項

- `/workspace/research/` は Vast.ai cloud copy で作成されたため **読み取り専用**
- 推論結果の保存先は `/workspace/results/` などroot所有のディレクトリを使用すること
- rcloneを使ったセットアップに切り替えることで書き込み可能なデータディレクトリを実現可能
