# HCM-CELLM ☁️⚡

**A Cloud-Edge Collaborative Large Language Model Inference Framework Based on Historical Context Matching**

This repository contains the official implementation, simulation environment, and the sampled dataset for the paper **"HCM-CELLM: A Cloud-Edge Collaborative Large Language Model Inference Framework Based on Historical Context Matching"** (Submitted to IEEE Internet of Things Journal).

## 📌 Overview
Deploying Large Language Models (LLMs) faces challenges like high inference latency, energy consumption, and privacy risks. HCM-CELLM introduces a novel framework that:
1. **Historical Context Matching**: Integrates Approximate Nearest Neighbor (ANN) search with a lightweight Transformer-based attention fine-tuning module.
2. **Sketch-based Parallel Paradigm**: Uses cloud LLMs for high-level semantic sketches and edge SLMs for parallel detail refinement.
3. **PPO-based Scheduling**: Employs Proximal Policy Optimization (PPO) with a trust-region caching mechanism to dynamically route tasks, minimizing latency and computational overhead.

## 📂 Dataset
For reproducibility, we provide the exact 100-query inference subset randomly sampled from the **MultiWOZ 2.2** dataset used in our simulations. 
- Location: `data/test_queries_100.json` (or `.parquet`)
- This dataset is utilized to test the matching efficacy and the PPO agent's convergence.

## ⚙️ Environment Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/HCM-CELLM.git
cd HCM-CELLM
