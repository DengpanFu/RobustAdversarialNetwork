# RobustAdversarialNetwork
A pytorch re-implementation for paper "[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)"

## Requirements
* pytorch>0.4
* torchvision
* tensorboardX

## Parameters
All the parameters are defined in config.py 
* `exp_name`: experiment name, will be used for construct output directory
* `snap_dir`: root directory to save snapshots, it works with `exp_name` to form a directory for a specific experiment

## Usage
### Training
```
python train.py
```

