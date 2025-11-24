# MoE Experiments: Mixture of Experts for Stock Price Forecasting

> **Core question**  
> I want to understand whether the **optimal time-series forecasting model depends on the volatility regime**.  
> Starting from a reproduction of **Time-MoE**, I find that it works well on structured electricity data but fails badly on highly volatile financial series such as Bitcoin, which motivates a **volatility-aware Mixture-of-Experts** that adapts model complexity (LSTM + Regression) to volatility.

---

## 1. Motivation & Research Question

Mixture-of-Experts (MoE) is very popular in large language models:  
only a subset of experts is activated for each token, saving computation while improving performance.

But for **time-series forecasting**, especially in **finance**, it is not clear that:

- the same MoE architecture will work equally well on
  - **stable / structured** series (e.g. electricity load, ETTh1), and  
  - **highly volatile, non-stationary** series (e.g. Bitcoin prices).

**🍀My research question is:**

> 🔍 *Does the best forecasting model depend on the volatility regime?*  
> If so, can we design a **volatility-aware MoE** that automatically adjusts model complexity to the asset’s volatility?

This repository contains:

1. A **reproduction and stress test** of a Time-MoE style model on:
   - ETTh1 electricity data (periodic, relatively stable)
   - Financial data (S&P 500 index, Bitcoin prices)
2. A **baseline LSTM** for comparison.
3. A **work-in-progress design** for a volatility-aware MoE combining:
   - a simple **regression / linear model** for stable data  
   - a **LSTM expert** for volatile data.

---

## 2. Project Overview

### 2.1 What I have done so far

- Implemented / adapted a **Time-MoE** architecture in PyTorch.
- Built a shared pipeline to:
  - load and preprocess time-series data (`dataset/`)
  - create sliding windows for multi-step forecasting
  - train models with a unified script (`main.py`)
  - evaluate and plot predictions (`eval_etth1.py`, `run_eval.py`).
- Ran experiments on:
  - **ETTh1** (electricity transformer temperature)
  - **S&P 500 index**
  - **Bitcoin price**

### 🔍2.2 Key empirical observation

- On **ETTh1** (periodic, structured, moderate volatility):  
  Time-MoE matches or slightly outperforms LSTM.  
- On **Bitcoin** (extremely volatile, non-stationary):  
  Time-MoE’s **MSE becomes very large**, and predictions are heavily over-smoothed / distorted compared to LSTM.

> ✅ Time-MoE makes sense for **non-sensitive / structured** series.  
> ❌ A naïve transplant to **ultra-volatile financial data** fails badly.

This leads to the new hypothesis:

> 💡 *“One-size-fits-all MoE doesn’t work in finance.  
> Different volatility regimes may favor different models.”*

---

## 3. Volatility-Aware MoE (Planned Extension)

To respond to the failure on Bitcoin, I propose a **volatility-aware MoE**:

- For **stable / low-volatility** series → favor a **Linear Regression / AR model**.
- For **volatile / high-volatility** series → favor a **nonlinear LSTM expert**.

A simple version could be:

1. Compute **rolling volatility** (e.g. 30-day std of log returns) for each asset / segment.
2. Classify regime:

   - `volatile` if σ > threshold (e.g. median or 0.025)  
   - `stable` otherwise.

3. Use a **mixture of two experts**:

   - Expert 1: Linear model / AR (LM)  
   - Expert 2: LSTM (RNN)

4. Static mixture weights (example):

   - For high-volatility assets/segments:  
     - w_RNN = 0.7, w_LM = 0.3  
   - For stable assets/segments:  
     - w_RNN = 0.3, w_LM = 0.7  

5. Future work: replace static weights with a **learnable gating network** that takes volatility-related features as input.

> This repo currently contains the **Time-MoE reproduction + volatility analysis**.  
> The volatility-aware MoE (LSTM + Regression) will be built on top of this codebase.

---

## 4. Repository Structure

The current structure (simplified) looks like this:

```text
.
├── dataset/              # Preprocessed time-series data (ETTh1, S&P, Bitcoin, etc.)
├── figures/              # Saved plots
│   ├── btc_eval.png      # Example: BTC predictions
│   └── etth1_eval.png    # Example: ETTh1 predictions
├── logs/                 # Training / evaluation logs
├── time_moe/             # Time-MoE checkpoints & configs (generic)
│   ├── btc_cpu_e3/       # Example: BTC run
│   ├── config.json
│   ├── generation_*.json
│   ├── model.safetensors
│   └── training_args.bin
├── time_moe_btc/         # Time-MoE runs specific to BTC (if used)
├── time_moe_etth1/       # Time-MoE runs specific to ETTh1 (if used)
├── timemoe_env/          # (Optional) environment or extra configs
├── tools/                # Utility scripts / helpers
├── demo_infer.py         # Minimal demo for running inference on a sequence
├── eval_etth1.py         # Evaluation on ETTh1 (JSONL loader + sliding windows + plotting)
├── main.py               # Main script for training / running Time-MoE
├── run_eval.py           # Generic evaluation script for other datasets (e.g. BTC)
├── torch_dist_run.py     # (Optional) distributed launch script
├── training_log.txt      # Example training log
├── requirements.txt      # Python dependencies
├── LICENSE
└── README.md             # You are here :)
