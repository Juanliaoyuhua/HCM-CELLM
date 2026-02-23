```markdown
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
* Location: `data/test_queries_100.json` (or `.parquet`)
* This dataset is utilized to test the matching efficacy and the PPO agent's convergence.

## ⚙️ Environment Setup

**1. Clone the repository:**
```bash
git clone https://github.com/[Your-Username]/HCM-CELLM.git
cd HCM-CELLM

```

**2. Install dependencies:**

```bash
pip install -r requirements.txt

```

*(Main dependencies include: `torch`, `transformers`, `chromadb`, `openai`, `bert_score`, `keybert`, `pandas`)*

**3. Prepare Local Models:**
Download the `all-mpnet-base-v2` embedding model from HuggingFace and place it in the `local_models/` directory. Update the path in `src/config.py` if necessary.

**4. Configure API Keys:**
The simulation uses third-party APIs for LLM generation. Set your environment variables before running:

```bash
export ARK_API_KEY="your_volcengine_ark_api_key_here"
# Note: Update edge model API keys in src/init_LLM.py

```

## 📊 Important Notes on Reproducibility

As discussed in the manuscript:

* **API Volatility**: Real-world API calls introduce network latency and throttling. We utilize a Moving Average mechanism to evaluate long-term trends.
* **Theoretical Latency**: The absolute latency metrics outputted by this simulation (e.g., ~24s) are theoretical values calculated based on our assumed edge/cloud mathematical capacity models (, ). The core algorithmic contribution is the relative latency reduction (~10.07%) driven by FLOPs savings.

## 📝 Citation

*(To be updated upon publication)*

```
