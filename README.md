# PruningNeuralNetworks
This repository provides the implementation of the method proposed in our paper "Pruning Deep Neural Networks using Partial Least Squares"

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://github.com/fchollet/keras) (Recommended version 2.1.2)
- [Tensorflow](https://www.tensorflow.org/) (Recommended version 1.3.0 or 1.9)
- [Python 3](https://www.python.org/)

## Quick Start
[main.py](main.py) provides an example of usage of our proposed pruning approach. In this example, we prune a simple convolutional neural network. 

## Parameters
Our method takes two parameters:
1. Number of pruning iterations (see line 243 in main.py)
2. Percentage of filters to be removed in each iteration (see line 244 in [main.py](main.py))
## Additional parameters (not recommended)
1. Number of components to the Partial Least Squares (see line 254 in [main.py](main.py))
2. Filter representation (see line 254 in [main.py](main.py)). The options are: 'max' and 'avg'. In addition, you can customize a pooling operation (i.e., max-pooling 2x2) to represent the filters (see line in [main.py](main.py))

## Limitations
The provided code is able to prune simple CNN architectures (VGG-based) since complex networks (i.e., with skip connections) are complicated to rebuild. On the other hand, you can employ our method to identify the potential filters to be removed and use [Keras-Surgeon] (https://github.com/BenWhetton/keras-surgeon) to rebuild the network.

### Results
Tables below show the comparison between our method with existing pruning methods. Negative values denote improvement regarding the original network. Please check our paper for more detailed results.

VGG16 on Cifar-10

|    Method   | Filter | FLOPs | Drop in Accuracy |
|:-----------:|:------:|:-----:|:----------------:|
|  Hu et al.  |  14.96 | 28.29 |       -0.66      |
|  Li et al.  |  37.12 | 34.00 |       -0.1       |
|    Huang    |  83.68 | 64.70 |        1.9       |
| Ours (it=1) |  9.99  | 23.21 |       -0.89      |
| Ours (it=5) |  40.93 | 67.28 |       -0.63      |
| Ours (it=9) |  68.63 | 90.69 |        1.5       |

ResNet56 on Cifar-10

|     Method     | Filter | FLOPs | Drop in Accuracy |
|:--------------:|:------:|:-----:|:----------------:|
|   Huang [10]   |    x   | 64.70 |        1.7       |
| Yu et al. [26] |    x   | 43.61 |       0.03       |
|   Ours(it=1)   |  4.34  |  7.95 |       -1.03      |
|   Ours(it=5)   |  17.60 | 31.48 |       -0.46      |
|   Ours(it=6)   |  24.49 | 48.01 |       0.34       |

Please cite our paper in your publications if it helps your research.
```bash
@article{Jordao:2018,
author    = {Artur Jordao,
Fernando Yamada and
William Robson Schwartz},
title     = {Pruning Deep Neural Networks using Partial Least Squares},
}
```
We would like to thank Ricardo Barbosa Kloss for the coffees and talks.
