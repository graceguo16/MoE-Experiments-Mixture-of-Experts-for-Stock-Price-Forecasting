# eval_etth1.py
import os, json, argparse, math
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

def load_jsonl(path):
    seqs = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            seq = obj.get("sequence", None)
            if seq is None:
                raise ValueError("Each line must be like: {\"sequence\": [...]} ")
            seqs.append(np.asarray(seq, dtype=float))
    if len(seqs) == 0:
        raise ValueError("Empty test file.")
    # 拼成一个长序列；若多行，就依次串接
    long_series = np.concatenate(seqs)
    return long_series

def sliding_windows(arr, max_len, pred_len, step):
    """
    生成 (context, target) 滑窗样本：
    context 长度<= max_len，target 长度= pred_len
    """
    samples = []
    i = 0
    while i + max_len + pred_len <= len(arr):
        ctx = arr[i : i + max_len]
        tgt = arr[i + max_len : i + max_len + pred_len]
        samples.append((ctx, tgt))
        i += step
    return samples

def standardize(x):
    mu = x.mean(axis=-1, keepdims=True)
    sig = x.std(axis=-1, keepdims=True)
    sig = np.clip(sig, 1e-6, None)
    return (x - mu) / sig, mu, sig

def inverse_standardize(xn, mu, sig):
    return xn * sig + mu

def mae(a, b): return float(np.mean(np.abs(a - b)))
def mse(a, b): return float(np.mean((a - b) ** 2))
def rmse(a, b): return math.sqrt(mse(a, b))

def plot_example(gt, pred, outpath, title="ETTh1 Forecast"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(9,4))
    plt.plot(gt, label="Actual")
    plt.plot(pred, label="TimeMoE Pred")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to finetuned model folder, e.g. logs/time_moe")
    ap.add_argument("--test", type=str, required=True, help="Path to test.jsonl")
    ap.add_argument("--max_length", type=int, default=256, help="context length (<= 4096)")
    ap.add_argument("--pred_len", type=int, default=24, help="forecast horizon")
    ap.add_argument("--step", type=int, default=1, help="stride for sliding windows")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"], help="device choice")
    args = ap.parse_args()

    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 读取测试序列
    series = load_jsonl(args.test)

    # 构造样本
    samples = sliding_windows(series, args.max_length, args.pred_len, args.step)
    if len(samples) == 0:
        raise ValueError("No test windows. Reduce pred_len or max_length, or use smaller step.")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device if device=="cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    if device == "cpu":
        model.to("cpu")

    all_tm_pred, all_gt, all_base = [], [], []
    # 评估
    for ctx_np, tgt_np in tqdm(samples, desc="Evaluating"):
        # 标准化（按窗口逐段标准化）
        ctx_norm, mu, sig = standardize(ctx_np[None, :])  # shape (1, L)
        ctx_t = torch.tensor(ctx_norm, dtype=torch.float32, device=model.device)

        with torch.no_grad():
            out = model.generate(ctx_t, max_new_tokens=args.pred_len)  # (1, L+H)
            pred_norm = out[:, -args.pred_len:].cpu().numpy()[0]

        pred = inverse_standardize(pred_norm, mu, sig)[0]  # 反标准化
        gt = tgt_np

        # 基线：last value 持续
        base = np.full_like(gt, fill_value=ctx_np[-1])

        all_tm_pred.append(pred)
        all_gt.append(gt)
        all_base.append(base)

    tm_pred = np.concatenate(all_tm_pred)
    gtruth = np.concatenate(all_gt)
    base_pred = np.concatenate(all_base)

    metrics = {
        "TimeMoE_MAE": mae(tm_pred, gtruth),
        "TimeMoE_MSE": mse(tm_pred, gtruth),
        "TimeMoE_RMSE": rmse(tm_pred, gtruth),
        "Baseline_MAE": mae(base_pred, gtruth),
        "Baseline_RMSE": rmse(base_pred, gtruth),
        "Windows": len(samples),
        "Horizon": args.pred_len,
        "MaxLen": args.max_length,
    }
    print(metrics)

    # 可视化第一窗
    fig_path = os.path.join("figures", "etth1_eval.png")
    # 拼一段：用第一窗的 gt 与 pred 作图
    plot_example(all_gt[0], all_tm_pred[0], fig_path, title=f"BTC Forecast (H={args.pred_len})")
    print(f"Saved figure -> {fig_path}")

if __name__ == "__main__":
    main()
