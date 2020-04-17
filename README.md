# Pruning Neural Networks
This repository provides the implementation of the method proposed in our paper "Pruning Deep Neural Networks using Partial Least Squares". The code in this repository is able to prune simple CNN architectures (i.e., VGG-based). To prune more sophisticated networks (i.e., ResNets), you can employ our method to identify potential filters to be removed and use [Keras-Surgeon](https://github.com/BenWhetton/keras-surgeon) to rebuild the network.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://github.com/fchollet/keras) (Recommended version 2.1.2)
- [Tensorflow](https://www.tensorflow.org/) (Recommended version 1.3.0 or 1.9)
- [Python 3](https://www.python.org/)

## Quick Start
[main.py](main.py) provides an example of our pruning approach. In this example, we prune a simple convolutional neural network. It should be mentioned that the network of this example is not the network used in our paper. Instead, we prefer to use this simple network due to the computational cost.

## Parameters
Our method takes two parameters:
1. Number of pruning iterations (see line 243 in [main.py](main.py))
2. Percentage of filters to be removed in each iteration (see line 244 in [main.py](main.py))
## Additional parameters (not recommended)
1. Number of components of Partial Least Squares (see line 254 in [main.py](main.py))
2. Filter representation (see line 254 in [main.py](main.py)). The options are: 'max' and 'avg'. In addition, you can customize a pooling operation (i.e., max-pooling 2x2) to represent the filters (see line 255 in [main.py](main.py))

## Results
Tables below show the comparison between our method with existing pruning methods. Negative values in accuracy denote improvement regarding the original network. Please check our paper for more detailed results.

ResNet56 on Cifar-10

|     Method     | FLOPs ↓ (%) | Accuracy ↓ (percentage points) |
|:--------------:|:-----:|:----------------:|
| [Yu et al.](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0601.pdf) |   43.61 |       0.03       |
| [He et al.](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.pdf) | 50.00 |  0.90       |
|   Ours(it=1)   |   7.09 | -0.60 |
|   Ours(it=5)   |  35.23 | -0.90 |
|   Ours(it=8)   |  52.56 |-0.62|

ResNet50 on ImageNet

|    Method   | FLOPs ↓ (%) | Accuracy ↓ (percentage points) |
|:-----------:|:-----:|:----------------:|
|  [Liu et al.](https://openreview.net/pdf?id=rJlnB3C5Ym)  |  36.70 |       1.01      |
|  [He et al.](https://arxiv.org/pdf/1808.06866.pdf)  |  41.80 |       8.27       |
|  [He et al.](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)   | 53.50 |        0.55       |
| Ours (it=1) | 6.13 |       -1.92      |
| Ours (it=5) | 27.45 |       -0.31      |
| Ours (it=10) | 44.50 |        1.01       |

Please cite our paper in your publications if it helps your research.
```bash
@inproceedings{Jordao:2019,
author    = {Artur Jordao,
Ricardo Kloss,
Fernando Yamada and
William Robson Schwartz},
title     = {Pruning Deep Neural Networks using Partial Least Squares},
booktitle = {British Machine Vision Conference (BMVC) Workshops: Embedded AI for Real-Time Machine Vision},
}
@article{Jordao::2020,
  author    = {Artur Jordao,
Fernando Yamada and
William Robson Schwartz},
  title     = {Deep network compression based on Partial Least Squares},
  journal   = {Neurocomputing},
  year      = {2020},
}
```
We would like to thank Maiko Lie for the coffees and talks.
