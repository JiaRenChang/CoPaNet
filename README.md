# Deep Competitive Pathway Network (CoPaNet)

This repository contains the code for CoPaNet introduced in the paper ["Deep Competitive Pathway Network"](arxiv link) by Jia-Ren Chang and Yong-Sheng Chen.

The code is built on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

## Introduction
CoPaNet is a network architecture where multiple pathways compete with each other. This network architecture yields a novel phenomenon which we called "pathway encoding". The pathway encoding means that the routing patterns of object can represent its semantic. The CoPaNet peforms state-of-the-art accuracy on CIFAR-10 and SVHN. On the large scale ILSVRC 2012 (ImageNet) dataset, CoPaNet achieves a similar accuracy as ResNet, but using less amount of parameters.

<img src="https://user-images.githubusercontent.com/11732099/30900569-130b4c76-a397-11e7-9c22-13410f9038a9.png" width="360">

Figure 1: The concept of pathway encoding. 

<img src="https://user-images.githubusercontent.com/11732099/30900881-357a3a50-a398-11e7-81b1-696e61edc064.png" width="360">
Figure 2: The pathway encoding on CIFAR-10 test set. 
