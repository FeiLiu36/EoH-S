# EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design

[![AAAI 2025](https://img.shields.io/badge/AAAI-2025-blue.svg)](https://openreview.net/forum?id=JiOY4d5ktq)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of [**"EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design"** ](https://arxiv.org/abs/2508.03082)accepted at **AAAI** 2025 as an **Oral** presentation.

[Paper](https://openreview.net/forum?id=JiOY4d5ktq) | [Project Page](#) | [Demo](#)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

**EoH-S (Evolution of Heuristic Set)** is a novel framework that leverages Large Language Models (LLMs) for automated heuristic design. Our approach evolves a small-sized set of complementary heuristics through iterative refinement, enabling superior cross-distribution performance across various algorithm design tasks.

### Key Features

- 🚀 **Automated Heuristic Design**: Leverages LLMs to automatically generate and evolve heuristics
- 🎯 **Set-based Evolution**: Maintains and evolves a diverse set of heuristics rather than a single solution
- 🔄 **Iterative Refinement**: Continuously improves heuristic quality through evolutionary strategies
- 📊 **Broad Applicability**: Tested on multiple combinatorial optimization tasks
- ⚡ **State-of-the-art Performance**: Achieves competitive results compared to existing methods

### Method Overview

![Figure 1: Overview of the EoH-S framework and Comparison to Existing LLM-driven AHD methods](https://github.com/FeiLiu36/EoH-S/raw/main/figures/figure1.png)

---

## 📁 Repository Structure

- **`code/`**: Source code for EoH-S and baseline methods, implemented using the LLM4AD platform
- **`datasets/`**: Training and testing datasets, along with instance generation scripts
- **`examples/`**: Running scripts and configurations for different optimization tasks
- **`heuristics/`**: Final heuristics designed by EoH-S and baseline methods
- **`results/`**: Detailed experimental results, including performance metrics and logs for all methods

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher and < 3.13
- pip package manager

### Setup Instructions

1. **Clone the repository**
   
   ```bash
   git clone https://github.com/yourusername/eohs.git
   cd eohs
   ```
   
2. **Install LLM4AD platform**
   ```bash
   cd code
   pip install .
   cd ..
   ```

3. **Set up API Endpoint, key, and LLM** in the running script (e.g., examples/obp_set/run_eohs.py)
   
   ```bash
       llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                      key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                      model='deepseek-v3',  # your llm, e.g., 'gpt-3.5-turbo'
                      timeout=60)
   ```
   

---

## 🚀 Quick Start

### Running EoH-S on Online Bin Packing

```bash
cd examples
python obp_set/run_eohs.py
```

The results will be saved in the `logs/` folder.

### Running Other Tasks

```bash
# For TSP
python tsp_set/run_eohs.py

# For other tasks
python <task_name>/run_eohs.py
```

### Customizing Experiments

```python
from llm4ad.task.optimization.online_bin_packing_set import OBPSEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eohs import EoHS,EoHSProfiler

def main():

    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='deepseek-v3',  # your llm, e.g., 'gpt-3.5-turbo'
                   timeout=60)

    task = OBPSEvaluation(
        timeout_seconds=120,
        dataset='./dataset_100_2k_128_5_80_training.pkl',
        return_list=True)

    method = EoHS(llm=llm,
                 profiler=EoHSProfiler(log_dir='logs/eohs', log_style='simple'),
                 evaluation=task,
                 max_sample_nums=2000,
                 max_generations=1000,
                 pop_size=10,
                 num_samplers=4,
                 num_evaluators=4,
                 debug_mode=False)

    method.run()

if __name__ == '__main__':
    main()

```

You can modify the configuration files in `examples/` to customize:

- Population size: pop_size

- Total number of sampled heuristics: max_sample_nums

- Number of parallel evaluations: num_samplers and num_evaluators

- Others: e.g., LLM type and timeout

  

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{liu2025eohs,
  title={EoH-S: Evolution of Heuristic Set using {LLM}s for Automated Heuristic Design},
  author={Fei Liu, Yilu Liu, Qingfu Zhang, Xialiang Tong, Mingxuan Yuan},
  booktitle={The Fortieth AAAI Conference on Artificial Intelligence},
  year={2025},
  url={https://openreview.net/forum?id=JiOY4d5ktq}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on the [LLM4AD](https://github.com/Optima-CityU/LLM4AD) platform
- Thanks to the AAAI 2025 reviewers for their valuable feedback

---

## 📧 Contact

For questions and feedback:
- Open an issue on [GitHub Issues](https://github.com/FeiLiu36/EoH-S/issues)
- Email: [fliu36-c@my.cityu.edu.hk]

---

