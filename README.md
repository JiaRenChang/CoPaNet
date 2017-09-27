# Deep Competitive Pathway Network (CoPaNet)

This repository contains the code for CoPaNet introduced in the paper ["Deep Competitive Pathway Network"](arxiv link) by Jia-Ren Chang and Yong-Sheng Chen.

The code is built on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

## Introduction
CoPaNet is a network architecture where multiple pathways compete with each other. This network architecture yields a novel phenomenon which we called "pathway encoding". The pathway encoding means that the routing patterns of object can represent its semantic. The CoPaNet peforms state-of-the-art accuracy on CIFAR-10 and SVHN. On the large scale ILSVRC 2012 (ImageNet) dataset, CoPaNet achieves a similar accuracy as ResNet, but using less amount of parameters.

<img src="https://user-images.githubusercontent.com/11732099/30900569-130b4c76-a397-11e7-9c22-13410f9038a9.png" width="360">

Figure 1: The concept of pathway encoding. 

<img src="https://user-images.githubusercontent.com/11732099/30900957-84c73e3c-a398-11e7-8672-df400e74c408.png" width="480">
Figure 2: The pathway encoding on CIFAR-10 test set. 

## Usage 
0. Install Torch and required dependencies like cuDNN. See the instructions [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md) for a step-by-step guide.
1. Clone this repo: ```https://github.com/JiaRenChang/CoPaNet.git```

As an example, the following command trains a CoPaNet with depth 164 on CIFAR-10:
```
th main.lua -netType CoPaNet -dataset cifar10 -batchSize 64 -nEpochs 300 -depth 164
``` 
As another example, the following command trains a CoPaNet with depth 26 on ImageNet:
```
th main.lua -netType CoPaNet -dataset imagenet -data [dataFolder] -batchSize 256 -nEpochs 100 -depth 16 -growthRate 32 -nGPU 4
``` 
Please refer to [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) for data preparation.

## Contact
followwar at gmail.com

followwar.cs00g at nctu.edu.tw   
Any discussions, suggestions and questions are welcome!
