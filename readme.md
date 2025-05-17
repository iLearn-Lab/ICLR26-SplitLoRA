# SplitLoRA: Balancing Stability and Plasticity in Continual Learning Through Gradient Space Splitting

## Environment
```
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
einops==0.7.0
ftfy==6.1.3
huggingface-hub==0.18.0
numpy==1.26.0
opencv-python==4.8.1.78
Pillow==10.0.1
regex==2023.12.25
scikit-image==0.22.0
scikit-learn==1.3.2
scipy==1.11.3
tqdm==4.66.1
```
These packages can be installed easily by
`pip install -r requirements.txt`

## Dataset preparation
### 1. Download the datasets and uncompress them:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet-R: https://github.com/hendrycks/imagenet-r
- DomainNet: https://ai.bu.edu/M3SDA/

### 2. Rearrange the directory structure:

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

## Training and evaluation

For three datasets: `python reproduce.py`
