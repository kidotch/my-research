"""
TimeLens-8B 推論スクリプト

使用方法:
    conda activate timelens
    python run_inference.py --base_dir /path/to/research

ディレクトリ構成（base_dir以下）:
    research/
    ├── datasets/          ← 動画クリップ
    ├── experiments/
    │   └── timelens/
    │       ├── plans/     ← テストデータJSON
    │       └── results/   ← 推論結果の保存先
    └── models/
        └── TimeLens-8B/   ← モデルウェイト

オプション:
    --base_dir      研究ルートディレクトリ（必須）
    --test_data     テストデータJSON（省略時: experiments/timelens/plans/test_data_20260419.json）
    --model         モデルパス（省略時: models/TimeLens-8B）
    --results_dir   結果保存先（省略時: experiments/timelens/results）
    --timelens_repo TimeLensリポジトリのパス（省略時: /workspace/TimeLens）
    --fps           動画サンプリングFPS（省略時: 2）
"""

import argparse
import json
import os
import sys
from datetime import datetime

import av
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",     required=True,
                        help="研究ルートディレクトリ（datasets/, experiments/, models/ を含む）")
    parser.add_argument("--test_data",    default=None,
                        help="テストデータJSON（base_dirからの相対パス or 絶対パス）")
    parser.add_argument("--model",        default=None,
                        help="モデルディレクトリ（base_dirからの相対パス or 絶対パス）")
    parser.add_argument("--results_dir",  default=None,
                        help="結果保存先（base_dirからの相対パス or 絶対パス）")
    parser.add_argument("--timelens_repo", default="/workspace/TimeLens",
                        help="TimeLensリポジトリのパス（デフォルト: /workspace/TimeLens）")
    parser.add_argument("--fps",          type=int, default=2)
    return parser.parse_args()


def resolve(base_dir, path, default_relative):
    """絶対パスならそのまま、Noneまたは相対パスならbase_dirと結合する"""
    if path is None:
        return os.path.join(base_dir, default_relative)
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def load_model(model_path):
    print(f"モデル読み込み中: {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    print("モデル読み込み完了\n")
    return model, processor


def run_inference(model, processor, video_path, query, fps=2):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 64 * 32 * 32,
                    "total_pixels": 14336 * 32 * 32,
                    "fps": fps,
                },
                {"type": "text", "text": GROUNDER_PROMPT.format(query)},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    videos, video_metadatas = zip(*videos)
    inputs = processor(
        text=[text],
        images=images,
        videos=list(videos),
        video_metadata=list(video_metadatas),
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    from timelens.utils import extract_time
    timestamps = extract_time(response)
    return response, timestamps


def get_duration(video_path):
    with av.open(video_path) as c:
        return round(float(c.duration) / 1e6, 3)


def calc_iou(pred, gt):
    if not pred:
        return 0.0
    p_start, p_end = pred[0]
    g_start, g_end = gt
    inter = max(0, min(p_end, g_end) - max(p_start, g_start))
    union = max(p_end, g_end) - min(p_start, g_start)
    return inter / union if union > 0 else 0.0


def main():
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)

    # パス解決
    model_path    = resolve(base_dir, args.model,       "models/TimeLens-8B")
    test_data_path = resolve(base_dir, args.test_data,  "experiments/timelens/plans/test_data_20260419.json")
    results_dir   = resolve(base_dir, args.results_dir, "experiments/timelens/results")

    print(f"base_dir:     {base_dir}")
    print(f"model:        {model_path}")
    print(f"test_data:    {test_data_path}")
    print(f"results_dir:  {results_dir}")
    print(f"timelens_repo:{args.timelens_repo}\n")

    # TimeLensリポジトリをパスに追加
    sys.path.insert(0, args.timelens_repo)

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(test_data_path, encoding="utf-8") as f:
        test_data = json.load(f)

    model, processor = load_model(model_path)

    results = []
    for i, item in enumerate(test_data):
        # clip_pathが相対パスならbase_dirと結合
        clip_path = item["clip_path"]
        if not os.path.isabs(clip_path):
            clip_path = os.path.join(base_dir, clip_path)

        print(f"[{i+1:2d}/{len(test_data)}] {item['id']}")
        print(f"       クリップ: {os.path.basename(clip_path)}")
        print(f"       クエリ:   {item['query']}")
        print(f"       正解:     {item['clip_time_start']:.3f}s - {item['clip_time_end']:.3f}s")

        duration = get_duration(clip_path)
        print(f"       尺:       {duration:.3f}s")

        response, pred_timestamps = run_inference(model, processor, clip_path, item["query"], args.fps)

        gt = (item["clip_time_start"], item["clip_time_end"])
        iou = calc_iou(pred_timestamps, gt)
        r1_03 = 1 if iou >= 0.3 else 0
        r1_05 = 1 if iou >= 0.5 else 0
        r1_07 = 1 if iou >= 0.7 else 0

        pred_str = f"{pred_timestamps[0][0]:.3f}s - {pred_timestamps[0][1]:.3f}s" if pred_timestamps else "検出なし"
        print(f"       予測:     {pred_str}")
        print(f"       IoU:      {iou:.3f}  R1@0.3={r1_03}  R1@0.5={r1_05}  R1@0.7={r1_07}")
        print()

        results.append({
            **{k: item[k] for k in ["id", "clip_path", "segment_id", "source_video",
                                     "label", "query", "clip_time_start", "clip_time_end"]},
            "clip_duration": duration,
            "response": response,
            "pred_timestamps": pred_timestamps,
            "iou": round(iou, 4),
            "r1_at_03": r1_03,
            "r1_at_05": r1_05,
            "r1_at_07": r1_07,
        })

    # 集計
    n = len(results)
    avg_iou   = sum(r["iou"]      for r in results) / n
    r1_03_avg = sum(r["r1_at_03"] for r in results) / n
    r1_05_avg = sum(r["r1_at_05"] for r in results) / n
    r1_07_avg = sum(r["r1_at_07"] for r in results) / n

    summary = {
        "model": model_path,
        "test_data": test_data_path,
        "timestamp": timestamp,
        "n_samples": n,
        "metrics": {
            "mIoU":   round(avg_iou,   4),
            "R1@0.3": round(r1_03_avg, 4),
            "R1@0.5": round(r1_05_avg, 4),
            "R1@0.7": round(r1_07_avg, 4),
        },
        "results": results,
    }

    # JSON保存
    json_path = f"{results_dir}/results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"JSON保存: {json_path}")

    # Markdown保存
    md_path = f"{results_dir}/results_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# TimeLens-8B 推論結果\n\n")
        f.write(f"- 実行日時: {timestamp}\n")
        f.write(f"- モデル: {model_path}\n")
        f.write(f"- サンプル数: {n}\n\n")
        f.write(f"## メトリクス\n\n")
        f.write(f"| mIoU | R1@0.3 | R1@0.5 | R1@0.7 |\n")
        f.write(f"|---|---|---|---|\n")
        f.write(f"| {avg_iou:.4f} | {r1_03_avg:.4f} | {r1_05_avg:.4f} | {r1_07_avg:.4f} |\n\n")
        f.write(f"## 詳細結果\n\n")
        f.write(f"| ID | ラベル | 尺(s) | 正解(s) | 予測(s) | IoU | @0.3 | @0.5 | @0.7 |\n")
        f.write(f"|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            gt_str   = f"{r['clip_time_start']:.2f}-{r['clip_time_end']:.2f}"
            pred_str = f"{r['pred_timestamps'][0][0]:.2f}-{r['pred_timestamps'][0][1]:.2f}" if r["pred_timestamps"] else "なし"
            f.write(f"| {r['id']} | {r['label']} | {r['clip_duration']:.1f} | {gt_str} | {pred_str} | {r['iou']:.3f} | {r['r1_at_03']} | {r['r1_at_05']} | {r['r1_at_07']} |\n")
    print(f"MD保存:   {md_path}")

    print(f"\n=== 最終メトリクス ===")
    print(f"mIoU:   {avg_iou:.4f}")
    print(f"R1@0.3: {r1_03_avg:.4f}")
    print(f"R1@0.5: {r1_05_avg:.4f}")
    print(f"R1@0.7: {r1_07_avg:.4f}")


if __name__ == "__main__":
    main()
