#!/usr/bin/env python3
"""
UniTime 推論スクリプト
timelens/run_inference.py と同じ plans JSON を受け取り、同形式で結果を保存する
"""

import argparse
import json
import os
import sys
import time
import types
from datetime import datetime, timedelta

import psutil
import torch


def _add_unitime_to_path(unitime_repo):
    if unitime_repo not in sys.path:
        sys.path.insert(0, unitime_repo)


def calc_iou(pred_windows, gt):
    if not pred_windows or pred_windows[0] == [-1, -1]:
        return 0.0
    ps, pe = pred_windows[0]
    gs, ge = gt
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def get_duration(video_path):
    import av
    with av.open(video_path) as c:
        return float(c.duration) / 1e6


def mem_stats():
    ram = psutil.virtual_memory()
    ram_used = (ram.total - ram.available) / 1024**3
    ram_total = ram.total / 1024**3
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_reserved() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"VRAM {vram_used:.1f}/{vram_total:.1f}GB  RAM {ram_used:.1f}/{ram_total:.1f}GB"
    return f"RAM {ram_used:.1f}/{ram_total:.1f}GB"


def load_model(base_model_path, finetune_path):
    from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
    from peft import PeftModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"ベースモデル読み込み中: {base_model_path}")
    model = Qwen2VLMRForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    if finetune_path:
        print(f"LoRAアダプター適用中: {finetune_path}")
        model = PeftModel.from_pretrained(model, finetune_path)
        model = model.merge_and_unload()

    model.eval()

    processor = Qwen2VLMRProcessor.from_pretrained(base_model_path)

    if torch.cuda.is_available():
        vram = torch.cuda.memory_reserved() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"モデル読み込み完了 — VRAM使用: {vram:.1f}GB / {vram_total:.1f}GB")

    return model, processor, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",    required=True,  help="研究ルートディレクトリ")
    parser.add_argument("--test_data",   required=True,  help="plans JSON（base_dirからの相対パスまたは絶対パス）")
    parser.add_argument("--unitime_repo",default="/workspace/UniTime", help="UniTimeリポジトリのパス")
    parser.add_argument("--base_model",  default="models/Qwen2-VL-7B-Instruct", help="Qwen2-VL-7Bベースモデル")
    parser.add_argument("--model",       default="models/UniTime",  help="UniTimeファインチューンウェイト")
    parser.add_argument("--results_dir", default="experiments/unitime/results")
    parser.add_argument("--fps",         type=int, default=2)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--nf_short",    type=int, default=128, help="この秒数以下の動画はmrモード（-1で無効）")
    parser.add_argument("--feat_folder", default="/tmp/unitime_features")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    _add_unitime_to_path(args.unitime_repo)

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    test_data_path = resolve(args.test_data)
    base_model_path = resolve(args.base_model)
    finetune_path   = resolve(args.model)
    results_dir     = resolve(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print(f"base_dir:     {base_dir}")
    print(f"base_model:   {base_model_path}")
    print(f"model:        {finetune_path}")
    print(f"test_data:    {test_data_path}")
    print(f"results_dir:  {results_dir}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(test_data_path, encoding="utf-8") as f:
        test_data = json.load(f)

    model, processor, device = load_model(base_model_path, finetune_path)

    from inference import run_inference as unitime_run_inference

    inference_args = types.SimpleNamespace(
        fps=args.fps,
        clip_length=args.clip_length,
        nf_short=args.nf_short,
        feat_folder=args.feat_folder,
        video_root=base_dir,
    )

    results = []
    elapsed_times = []

    for i, item in enumerate(test_data):
        video_path = item["video_path"]
        if not os.path.isabs(video_path):
            video_path = os.path.join(base_dir, video_path)

        duration = item.get("duration") or get_duration(video_path)

        print(f"[{i+1:2d}/{len(test_data)}] {item['id']}")
        print(f"       動画:     {os.path.basename(video_path)}")
        print(f"       クエリ:   {item['query']}")
        print(f"       正解:     {item['time_start']:.3f}s - {item['time_end']:.3f}s")
        print(f"       尺:       {duration:.3f}s")
        print(f"       [{mem_stats()}]")

        data = {
            "qid":      i,
            "id":       item["id"],
            "duration": duration,
            "video_path": video_path,
            "annos":    [{"query": item["query"], "window": [[item["time_start"], item["time_end"]]]}],
        }

        t0 = time.time()
        unitime_results = unitime_run_inference(model, processor, data, inference_args, device)
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        elapsed_times.append(elapsed)

        avg_elapsed = sum(elapsed_times) / len(elapsed_times)
        remaining = avg_elapsed * (len(test_data) - (i + 1))
        eta_str = str(timedelta(seconds=int(remaining)))
        print(f"       推論時間: {elapsed:.1f}s  ETA {eta_str}")

        pred_windows = unitime_results[0]["pred_relevant_windows"] if unitime_results else [[-1, -1]]
        gt = (item["time_start"], item["time_end"])
        iou = calc_iou(pred_windows, gt)
        r1_03 = 1 if iou >= 0.3 else 0
        r1_05 = 1 if iou >= 0.5 else 0
        r1_07 = 1 if iou >= 0.7 else 0

        pred_str = f"{pred_windows[0][0]:.3f}s - {pred_windows[0][1]:.3f}s" \
            if pred_windows and pred_windows[0] != [-1, -1] else "検出なし"
        print(f"       予測:     {pred_str}")
        print(f"       IoU:      {iou:.3f}  R1@0.3={r1_03}  R1@0.5={r1_05}  R1@0.7={r1_07}")
        print()

        results.append({
            **{k: item[k] for k in ["id", "video_path", "label", "query", "time_start", "time_end"]},
            "duration":       duration,
            "pred_timestamps": pred_windows,
            "iou":     round(iou, 4),
            "r1_at_03": r1_03,
            "r1_at_05": r1_05,
            "r1_at_07": r1_07,
        })

    n = len(results)
    avg_iou   = sum(r["iou"]      for r in results) / n
    r1_03_avg = sum(r["r1_at_03"] for r in results) / n
    r1_05_avg = sum(r["r1_at_05"] for r in results) / n
    r1_07_avg = sum(r["r1_at_07"] for r in results) / n

    summary = {
        "model":     finetune_path,
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

    json_path = f"{results_dir}/results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"JSON保存: {json_path}")

    md_path = f"{results_dir}/results_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# UniTime 推論結果\n\n")
        f.write(f"- 実行日時: {timestamp}\n")
        f.write(f"- モデル: {finetune_path}\n")
        f.write(f"- サンプル数: {n}\n\n")
        f.write(f"## メトリクス\n\n")
        f.write(f"| mIoU | R1@0.3 | R1@0.5 | R1@0.7 |\n")
        f.write(f"|---|---|---|---|\n")
        f.write(f"| {avg_iou:.4f} | {r1_03_avg:.4f} | {r1_05_avg:.4f} | {r1_07_avg:.4f} |\n\n")
        f.write(f"## 詳細結果\n\n")
        f.write(f"| ID | ラベル | 尺(s) | 正解(s) | 予測(s) | IoU | @0.3 | @0.5 | @0.7 |\n")
        f.write(f"|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            gt_str   = f"{r['time_start']:.2f}-{r['time_end']:.2f}"
            pred = r["pred_timestamps"]
            pred_str = f"{pred[0][0]:.2f}-{pred[0][1]:.2f}" if pred and pred[0] != [-1, -1] else "なし"
            f.write(f"| {r['id']} | {r['label']} | {r['duration']:.1f} | {gt_str} | {pred_str} | {r['iou']:.3f} | {r['r1_at_03']} | {r['r1_at_05']} | {r['r1_at_07']} |\n")
    print(f"MD保存:   {md_path}")

    print(f"\n=== 最終メトリクス ===")
    print(f"mIoU:   {avg_iou:.4f}")
    print(f"R1@0.3: {r1_03_avg:.4f}")
    print(f"R1@0.5: {r1_05_avg:.4f}")
    print(f"R1@0.7: {r1_07_avg:.4f}")


if __name__ == "__main__":
    main()
