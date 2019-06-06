# Pruning Neural Networks
This repository provides the implementation of the method proposed in our paper "Pruning Deep Neural Networks using Partial Least Squares"

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

## Limitations
The provided code is able to prune simple CNN architectures (VGG-based) since complex networks (i.e., with skip connections) are complicated to rebuild. On the other hand, you can employ our method to identify the potential filters to be removed and use [Keras-Surgeon](https://github.com/BenWhetton/keras-surgeon) to rebuild the network.

### Results
Tables below show the comparison between our method with existing pruning methods. Negative values in accuracy denote improvement regarding the original network. Please check our paper for more detailed results.

VGG16 on Cifar-10

|    Method   | FLOPs ↓ (%) | Accuracy ↓ (percentage points) |
|:-----------:|:-----:|:----------------:|
|  [Hu et al.](https://arxiv.org/abs/1607.03250)  |  28.29 |       -0.66      |
|  [Li et al.](https://arxiv.org/abs/1608.08710)  |  34.00 |       -0.10       |
|    [Huang](https://ieeexplore.ieee.org/document/8354187)   | 64.70 |        1.90       |
| Ours (it=1) | 23.13 |       -0.89      |
| Ours (it=5) | 67.25 |       -0.63      |
| Ours (it=9) | 90.66 |        1.50       |

ResNet56 on Cifar-10

|     Method     | FLOPs ↓ (%) | Accuracy ↓ (percentage points) |
|:--------------:|:-----:|:----------------:|
| [Yu et al.](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0601.pdf) |   43.61 |       0.03       |
| [He et al.](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.pdf) | 50.00 |  0.90       |
|   Ours(it=1)   |   7.09 | -0.60 |
|   Ours(it=5)   |  35.23 | -0.90 |
|   Ours(it=8)   |  52.56 |-0.62|

Please cite our paper in your publications if it helps your research.
```bash
@article{Jordao:2018,
author    = {Artur Jordao,
Ricardo Kloss,
Fernando Yamada and
William Robson Schwartz},
title     = {Pruning Deep Neural Networks using Partial Least Squares},
journal = {ArXiv e-prints},
eprint = {1810.07610},
}
```
We would like to thank Maiko Lie for the coffees and talks.
