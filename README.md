# RectifiedFlow

This is the official implementation of paper 
## [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) 
by *Xingchao Liu, Chengyue Gong, Qiang Liu* from UT Austin

## Introduction

Rectified Flow is a novel method for learning transport maps between two distributions $\pi_0$ and $\pi_1$, by connecting **straight paths** between the samples and learning an ODE model.

Then, by a **reflow** operation, we iteratively straighten the ODE trajectories to eventually achieve **one-step** generation.

An introductory website can be found [here](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html) and the main idea is illustrated in the following figure:

![](assets/intro_two_gauss.png)

Rectified Flow can be applied to both generative modeling and unsupervised domain transfer, as shown in the following figure:

![](assets/intro_rf.png)

## Interactive Colab notebooks

## Citation

