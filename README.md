# Bag of Tricks for Fully Test-Time Adaptation

This repository contains the code used for [Bag of Tricks for Fully Test-Time Adaptation 🔗](https://arxiv.org/pdf/2310.02416.pdf) by
Saypraseuth Mounsaveng, Florent Chiaroni, Malik Boudiaf, Marco Pedersoli and Ismail Ben Ayed **(WACV 2024)**.

# Code
The code is mainly inspired from [SAR 🔗](https://github.com/mr-eggplant/SAR) and adapted with additional methods and tricks.

**Links to the source code of the methods mentioned in the article**:
The code used in this paper was mostly inspired by the
| Method | Code Link |
| ---|:---|
| Tent | https://github.com/DequanWang/tent |
| SAR | https://github.com/mr-eggplant/SAR |
| Delta | https://github.com/bwbwzhao/DELTA |
| DUA |  https://github.com/jmiemirza/DUA |
| Hebbian | n/a |

# Preparation

**Links to the weights of the pretrained models mentioned in the paper**:
| Architecture | Code Link |
| ---|:---|
| ResNet50-BN | https://download.pytorch.org/models/resnet50--9c8e357.pth |
| ResNet50-GN | timm |
| ResNet-101 | https://github.com/Albert0147/NRC\_SFDA |
| VitBase-LN | timm |
| WRN28-10 | RobustBench |
| WRN40-2 | RobustBench |
| SVHN model | Pytorch-Playground |

**Installation**:

Packages to install:

- [RobustBench](https://pytorch.org/)
- [timm](https://github.com/rwightman/pytorch-image-models)


**Data preparation**:

This repository contains code for evaluation on different datasets. Here are the links to download them:

[ImageNet-C 🔗](https://zenodo.org/record/2235448#.YpCSLxNBxAc).
[ImageNet-Sketch 🔗](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA).
[ImageNet-Rendition 🔗](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar).
[VisDA-C 2017 🔗](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification).
[CIFAR10 🔗](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).
[CIFAR100 🔗](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz).
[MNIST 🔗](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/download?datasetVersionNumber=1).
[MNIST-M 🔗](https://www.kaggle.com/datasets/profsoft/mnistm-dataset/download?datasetVersionNumber=1).
[USPS 🔗](https://git-disl.github.io/GTDLBench/datasets/usps_dataset/).

## Example: Adapting a pre-trained model on ImageNet-C (Corruption).

**Usage**:

```
python3 main.py --data_corruption /path/to/imagenet-c --exp_type [normal/bs1/mix_shifts/label_shifts] --method [no_adapt/tent/delta/sar] --model [resnet50_bn_torch/resnet50_gn_timm/vitbase_timm] --test_batch_size 16 --output /output/dir
```

'--exp_type' is choosen from:

- 'normal' means the same test setting to prior mild data stream in Tent and EATA

- 'bs1' means single sample adaptation, only one sample comes each time-step

- 'mix_shifts' conducts exps over the mixture of 15 corruption types in ImageNet-C

- 'label_shifts' means exps under online imbalanced label distribution shifts. Moreover, imbalance_ratio indicates the imbalance extent


 ## Experimental results

Please check our [PAPER 🔗](https://arxiv.org/pdf/2310.02416.pdf) for experimental results.


## Correspondence

For any questions or comments, please contact Saypraseuth Mounsaveng by [saypraseuth.mounsaveng.1 at etsmtl.net] .  📬


## Citation
If you think our work is helpful for the community, please consider citing it:
```
@InProceedings{Mounsaveng_2024_WACV,
    author    = {Mounsaveng, Saypraseuth and Chiaroni, Florent and Boudiaf, Malik and Pedersoli, Marco and Ben Ayed, Ismail},
    title     = {Bag of Tricks for Fully Test-Time Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024}
}
```
