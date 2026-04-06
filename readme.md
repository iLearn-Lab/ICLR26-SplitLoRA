# ICLR26-P-M

[ICLR 2026] Official Implementation for **SplitLoRA: Balancing Stability and Plasticity in Continual Learning Through Gradient Space Splitting**

## Authors

**Haomiao Qiu**<sup>1,2</sup>, **Miao Zhang**<sup>1</sup>\*, **Ziyue Qiao**<sup>2</sup>\*, **Weili Guan**<sup>1</sup>, **Min Zhang**<sup>1</sup>, **Liqiang Nie**<sup>1</sup>

<sup>1</sup> `Harbin Institute of Technology (Shenzhen)`  
<sup>2</sup> `Great Bay University`  
\* Corresponding author

## Links

- **Paper**: [`Paper Link`](https://openreview.net/forum?id=Zm1hjXxRQV)
- **Code Repository**: [`GitHub`](https://github.com/iLearn-Lab/NeurIPS25-SplitLoRA)


---

## Updates

- [05/2025] Initial release

---

## Introduction

 We  present SplitLoRA, a method for continual learning that combines orthogonal projection with LoRA. It improves the balance between plasticity and stability by effectively mitigating interference between new and old tasks. This repository provides the official implementation, train and evaluation scripts.

---


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iLearn-Lab/ICLR26-SplitLoRA.git
cd ICLR26-SplitLoRA
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---
### 4. Dataset preparation
#### Download the datasets and uncompress them:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet-R: https://github.com/hendrycks/imagenet-r
- DomainNet: https://ai.bu.edu/M3SDA/

#### Rearrange the directory structure:

Directory structure for three datasets:
```
DATA_ROOT
    |- train
    |    |- class_folder_1
    |    |    |- image_file_1
    |    |    |- image_file_2
    |    |- class_folder_2
    |         |- image_file_2
    |         |- image_file_3
    |- val
         |- class_folder_1
         |    |- image_file_5
         |    |- image_file_6
         |- class_folder_2
              |- image_file_7
              |- image_file_8
```
We provide the scripts `split_[dataset].py` in the `tools` folder to rearange the directory structure.
Please change the `root_dir` in each script to the path of the uncompressed dataset.


---

## Usage

For three datasets: `python reproduce.py`


---


## Citation


```bibtex
@article{qiu2025splitlora,
  title={SplitLoRA: Balancing Stability and Plasticity in Continual Learning Through Gradient Space Splitting},
  author={Qiu, Haomiao and Zhang, Miao and Qiao, Ziyue and Guan, Weili and Zhang, Min and Nie, Liqiang},
  journal={arXiv preprint arXiv:2505.22370},
  year={2025}
}

```

---

## Acknowledgement

- Thanks to our supervisor and collaborators for valuable support.
- The code is developed based on https://github.com/zugexiaodui/VPTinNSforCL! We sincerely thank the authors for open-sourcing their code.

---

## License

This project is released under the Apache License 2.0.



