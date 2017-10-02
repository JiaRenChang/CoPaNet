# Deep Competitive Pathway Network (CoPaNet)

This repository contains the code for CoPaNet introduced in the paper ["Deep Competitive Pathway Network"](https://arxiv.org/abs/1709.10282) by Jia-Ren Chang and Yong-Sheng Chen.

This paper is accepted by Asian Conference on Machine Learning (ACML) 2017. 

The code is built on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

## Introduction
CoPaNet is a network architecture where multiple pathways compete with each other. This network architecture yields a novel phenomenon which we called "pathway encoding". The pathway encoding means that the routing patterns of features can represent object semantic. The CoPaNet peforms state-of-the-art accuracy on CIFAR-10 and SVHN. On the large scale ILSVRC 2012 (ImageNet) dataset, CoPaNet achieves a similar accuracy as ResNet, but using less amount of parameters.

<img src="https://user-images.githubusercontent.com/11732099/30900569-130b4c76-a397-11e7-9c22-13410f9038a9.png" width="360">

Figure 1: The concept of pathway encoding. 

<img src="https://user-images.githubusercontent.com/11732099/30900957-84c73e3c-a398-11e7-8672-df400e74c408.png" width="480">
Figure 2: The pathway encoding on CIFAR-10 test set. 

## Usage 
0. Install Torch and required dependencies like cuDNN. See the instructions [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md) for a step-by-step guide.
1. Clone this repo: ```https://github.com/JiaRenChang/CoPaNet.git```

We also provide our implementation of "CMaxTable".  
It runs above 2x faster than naive implementation in Torch's nn.

As an example, the following command trains a CoPaNet with depth 164 on CIFAR-10:
```
th main.lua -netType CoPaNet -dataset cifar10 -batchSize 128 -nEpochs 300 -depth 164
``` 
As another example, the following command trains a CoPaNet with depth 26 on ImageNet:
```
th main.lua -netType CoPaNet -dataset imagenet -data [dataFolder] -batchSize 256 -nEpochs 100 -depth 26 -nGPU 4
``` 
Please refer to [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) for data preparation.

## Contact
followwar at gmail.com  
Any discussions, suggestions and questions are welcome!
