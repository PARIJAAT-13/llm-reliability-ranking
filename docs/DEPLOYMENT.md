# Reproducible Multi-Hardware & Multi-Node Deployment Specification

This guide outlines step-by-step instructions for deploying and running the **LLM Reliability Ranking Framework** across heterogeneous hardware platforms (Local Workstations, Multi-GPU Cloud Clusters, and Edge Devices).

---

## 💻 1. Local Workstation Profile (`Local_x86_CPU_RAM16GB`)

### Target Specifications
- **CPU**: 16 Logical Cores (x86_64 / AMD64)
- **RAM**: 16 GB System RAM
- **OS**: Windows 10/11 or Ubuntu 22.04 LTS
- **Inference Runtime**: Ollama (v0.3.0+)

### Setup Commands
```bash
# Clone repository
git clone https://github.com/parijaat/llm-reliability-ranking.git
cd llm-reliability-ranking

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install framework
pip install -e .

# Start Ollama service and pull models
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull gemma2:9b

# Run full experiment study
python scripts/run_large_scale_experiment.py --config configs/full_experiment_config.json --output-dir results/full_study
```

---

## ☁️ 2. Cloud Multi-GPU Cluster Profile (`Cloud_NVIDIA_A100_80GB`)

### Target Specifications
- **Nodes**: 8x NVIDIA A100-SXM4-80GB (640GB aggregate VRAM)
- **Host RAM**: 256 GB DDR5
- **Inference Runtime**: vLLM or llama.cpp server
- **OS**: Ubuntu 22.04 LTS with CUDA 12.2

### Docker / Container Deployment
```bash
# Build production container image
docker build -t llm-reliability:v1.0 .

# Launch container with GPU access
docker run --gpus all --ipc=host -v $(pwd)/results:/app/results -p 8000:8000 llm-reliability:v1.0 \
  python scripts/run_large_scale_experiment.py \
  --config configs/full_experiment_config.json \
  --output-dir results/cloud_study
```

---

## 🍏 3. Apple Silicon Edge Profile (`Edge_Apple_M3_32GB`)

### Target Specifications
- **CPU/GPU**: Apple M3 Max with Unified Memory Architecture (32 GB)
- **OS**: macOS Sonoma 14.5+
- **Inference Runtime**: Ollama (Metal API accelerated)

### Setup Commands
```bash
# Install Ollama via Homebrew
brew install ollama
ollama serve &

# Run benchmarking execution script
python scripts/run_large_scale_experiment.py --config configs/full_experiment_config.json --output-dir results/edge_study
```
