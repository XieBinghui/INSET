# INSET

This repo contains PyTorch implementation of the paper "Enhancing Neural Subset Selection: Integrating Background Information Into Set Representations.".

## Installation

Please ensure that:

- Python >= 3.7
- PyTorch >= 1.8.0

## Experiments

### Product Recommendation

To run on the Amazon baby registry dataset
```
python main.py equivset --train --cuda --symmetry --data_name amazon --amazon_cat <category_name>
```
`category_name` is chosen in ['toys', 'furniture', 'gear', 'carseats', 'bath', 'health', 'diaper', 'bedding', 'safety', 'feeding', 'apparel', 'media'].

### Set Anomaly Detection

To run on the CelebA dataset
```
python main.py equivset --train --cuda --symmetry --data_name celeba
```

## Reference Code
Our code is built upon the framework provided by https://github.com/SubsetSelection/EquiVSet/tree/main.


## Citation

If you find our paper and repo useful, please consider to cite our paper:
```
@inproceedings{
xie2024enhancing,
title={Enhancing Neural Subset Selection: Integrating Background Information into Set Representations},
author={Binghui Xie and Yatao Bian and Kaiwen Zhou and Yongqiang Chen and Peilin Zhao and Bo Han and Wei Meng and James Cheng},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=eepoE7iLpL}
}
```