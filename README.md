# standalone-center-loss
Evaluating the effectiveness of using standalone center loss.

NOTE: Some of the code are following KaiyangZhou's [implementation](https://github.com/KaiyangZhou/pytorch-center-loss) of center loss!!!

## Introduction

In [Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016](https://ydwen.github.io/papers/WenECCV16.pdf), the author proposed to train CNNs under the joint supervision of the softmax loss and center loss, with a hyper parameter to balance the two supervision loss. The softmax loss forces the deep features of different classes staying apart. The center loss efficiently pulls the deep features of the same class to their centers. With the joint supervision, not only the inter-class features differences are enlarged, but also the intra-class features variations are reduced. Here, I'd like to explore alternatives to the softmax loss term. Intuitively, the overall loss function should pull the intra-class samples together, and push the inter-class samples away. Following this simple idea, the intra loss is defined the same as the center loss, whereas, the inter loss is defined by the negative distance of two samples of different classes. This way, reducing the inter loss becomes directly enlarging the distance across different classes. However, this inter loss term has no global minima, which can go down rapidly to ![](http://latex.codecogs.com/png.latex?-\infty), and turn the weights of the model to `nan`. A way to address this problem is to truncate the inter loss by a margin.  The distance across the deeply learned features should reflect the relationship across the dataset. Thus, introducing the manually selected margin may not maintain the relationship. Instead of truncating the inter loss by a margin, here the problem is addressed by applying logarithmic function to the inter loss. Theoretically, the log inter loss can also go toward ![](http://latex.codecogs.com/png.latex?-\infty), but it slows down declining as the value decrease. Furthermore, combining with the intra loss can also help to learn stable features since increasing the distance across different classes may also increasing the intra loss, reducing the intra loss helps to prevent distances across different classes going toward infinity. By doing the gradient analysis, applying logarithmic function to the inter loss is in fact equivalent to adding different weights to the inter loss, which forces the model to focus more on the hard samples.

The loss function used in this repo is:

![](http://latex.codecogs.com/png.latex?\mathcal{L}=w_{intra}*\mathcal{L}_{intra}+w_{inter}*\mathcal{L}_{inter})

![](http://latex.codecogs.com/png.latex?\mathcal{L}_{intra}=\frac{\sum_{i=1}^{m}\left\|\boldsymbol{x}_{i}-\boldsymbol{c}_{y_{i}}\right\|_{2}^{2}}{m})

![](http://latex.codecogs.com/png.latex?\mathcal{L}_{inter}=-\frac{\sum_{i=1}^{m}\sum_{j=1}^{m}I(y_i,y_j)*\log(\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|_{2}^{2})}{m})

where ![](http://latex.codecogs.com/png.latex?I(y_i,y_j)=0)if ![](http://latex.codecogs.com/png.latex?y_i=y_j) otherwise ![](http://latex.codecogs.com/png.latex?I(y_i,y_j)=1), ![](http://latex.codecogs.com/png.latex?w_{intra}) and ![](http://latex.codecogs.com/png.latex?w_{inter}) are hyper parameters to balance the two supervision loss.

## Experiment

|    loss    |    dataset    | feat_dim |  acc  |  lr  | epochs | batch_size | weight_cent | weight_intra | weight_inter |
| :--------: | :-----------: | :------: | :---: | :--: | :----: | :--------: | :---------: | :----------: | :----------: |
|    xent    |     mnist     |    2     | 0.991 | 0.01 |  100   |    128     |      /      |      /       |      /       |
|    cent    |     mnist     |    2     | 0.990 | 0.01 |  100   |    128     |     1.0     |      /       |      /       |
| standalone |     mnist     |    2     | 0.994 | 0.01 |  100   |    128     |      /      |     1.0      |     0.1      |
|    xent    |     mnist     |   128    | 0.994 | 0.01 |  100   |    128     |      /      |      /       |      /       |
|    cent    |     mnist     |   128    | 0.996 | 0.01 |  100   |    128     |     1.0     |      /       |      /       |
| standalone |     mnist     |   128    | 0.996 | 0.01 |  100   |    128     |      /      |     1.0      |     0.1      |
|    xent    | fashion-mnist |    2     | 0.913 | 0.01 |  100   |    128     |      /      |      /       |      /       |
|    cent    | fashion-mnist |    2     | 0.913 | 0.01 |  100   |    128     |     1.0     |      /       |      /       |
| standalone | fashion-mnist |    2     | 0.921 | 0.01 |  100   |    128     |      /      |     1.0      |     0.1      |
|    xent    | fashion-mnist |   128    | 0.926 | 0.01 |  100   |    128     |      /      |      /       |      /       |
|    cent    | fashion-mnist |   128    | 0.932 | 0.01 |  100   |    128     |     1.0     |      /       |      /       |
| standalone | fashion-mnist |   128    | 0.922 | 0.01 |  100   |    128     |      /      |     1.0      |     0.1      |
|    xent    |   cifar-10    |    2     | 0.815 | 0.01 |  100   |    128     |      /      |      /       |      /       |
|    cent    |   cifar-10    |    2     | 0.775 | 0.01 |  100   |    128     |     1.0     |      /       |      /       |
| standalone |   cifar-10    |    2     | 0.787 | 0.01 |  100   |    128     |      /      |     1.0      |     0.1      |
|    xent    |   cifar-10    |   128    | 0.866 | 0.01 |  100   |    128     |      /      |      /       |      /       |
|    cent    |   cifar-10    |   128    | 0.858 | 0.01 |  100   |    128     |     1.0     |      /       |      /       |
| standalone |   cifar-10    |   128    | 0.806 | 0.01 |  100   |    128     |      /      |     1.0      |     0.1      |

## Visualization

|    loss    |    dataset    | feat_dim |                        train                        |                        val                        |
| :--------: | :-----------: | :------: | :-------------------------------------------------: | :-----------------------------------------------: |
|    xent    |     mnist     |    2     |        ![](imgs/xent-mnist-feat=2/train.gif)        |        ![](imgs/xent-mnist-feat=2/val.gif)        |
|    cent    |     mnist     |    2     |        ![](imgs/cent-mnist-feat=2/train.gif)        |        ![](imgs/cent-mnist-feat=2/val.gif)        |
| standalone |     mnist     |    2     |     ![](imgs/standalone-mnist-feat=2/train.gif)     |     ![](imgs/standalone-mnist-feat=2/val.gif)     |
|    xent    | fashion-mnist |    2     |    ![](imgs/xent-fashion-mnist-feat=2/train.gif)    |    ![](imgs/xent-fashion-mnist-feat=2/val.gif)    |
|    cent    | fashion-mnist |    2     |    ![](imgs/cent-fashion-mnist-feat=2/train.gif)    |    ![](imgs/cent-fashion-mnist-feat=2/val.gif)    |
| standalone | fashion-mnist |    2     | ![](imgs/standalone-fashion-mnist-feat=2/train.gif) | ![](imgs/standalone-fashion-mnist-feat=2/val.gif) |
|    xent    |   cifar-10    |    2     |       ![](imgs/xent-cifar10-feat=2/train.gif)       |       ![](imgs/xent-cifar10-feat=2/val.gif)       |
|    cent    |   cifar-10    |    2     |       ![](imgs/cent-cifar10-feat=2/train.gif)       |       ![](imgs/cent-cifar10-feat=2/val.gif)       |
| standalone |   cifar-10    |    2     |    ![](imgs/standalone-cifar10-feat=2/train.gif)    |    ![](imgs/standalone-cifar10-feat=2/val.gif)    |

|    loss    |    dataset    | feat_dim |                         train                         |                         val                         |
| :--------: | :-----------: | :------: | :---------------------------------------------------: | :-------------------------------------------------: |
|    xent    |     mnist     |   128    |        ![](imgs/xent-mnist-feat=128/train.png)        |        ![](imgs/xent-mnist-feat=128/val.png)        |
|    cent    |     mnist     |   128    |        ![](imgs/cent-mnist-feat=128/train.png)        |        ![](imgs/cent-mnist-feat=128/val.png)        |
| standalone |     mnist     |   128    |     ![](imgs/standalone-mnist-feat=128/train.png)     |     ![](imgs/standalone-mnist-feat=128/val.png)     |
|    xent    | fashion-mnist |   128    |    ![](imgs/xent-fashion-mnist-feat=128/train.png)    |    ![](imgs/xent-fashion-mnist-feat=128/val.png)    |
|    cent    | fashion-mnist |   128    |    ![](imgs/cent-fashion-mnist-feat=128/train.png)    |    ![](imgs/cent-fashion-mnist-feat=128/val.png)    |
| standalone | fashion-mnist |   128    | ![](imgs/standalone-fashion-mnist-feat=128/train.png) | ![](imgs/standalone-fashion-mnist-feat=128/val.png) |
|    xent    |   cifar-10    |   128    |       ![](imgs/xent-cifar10-feat=128/train.png)       |       ![](imgs/xent-cifar10-feat=128/val.png)       |
|    cent    |   cifar-10    |   128    |       ![](imgs/cent-cifar10-feat=128/train.png)       |       ![](imgs/cent-cifar10-feat=128/val.png)       |
| standalone |   cifar-10    |   128    |    ![](imgs/standalone-cifar10-feat=128/train.png)    |    ![](imgs/standalone-cifar10-feat=128/val.png)    |

## Evaluation

Since no softmax loss is used in this implementation, using `argmax` to compute the accuracy is infeasible. Here, the evaluation is done by assigning each sample to the nearest center.

## Installation

1. install [PyTorch](https://pytorch.org/)
2. run the following command:

```shell
pip3 install -r requirements.txt
```

## Discussion

If you watch the train log of both standalone center loss and center loss with softmax loss, you‘ll find that the testset accuracy is much closer to the trainset accuracy under the supervision of standalone center loss than the center loss with softmax loss, does this mean that standalone center loss is less overfit prone, or it's just because the standalone center loss is inferior to the center loss with softmax loss?

As the gifs shown above, both two loss function really do a good job on mnist dataset, the reason for this could be that mnist is a fairly simple dataset. Therefore, next time you devise your new loss function, try it first on other dataset like fashion-mnist or cifar10 rather than on mnist would be a good choice, and further verify its effectiveness on mnist later.

Is it possible to build a much powerful model to turn a complicated dataset to a simple one, then the standalone center loss can learn a good feature for that dataset?

## Misc

Alternatives I've tried are summarized as follows:

1. **directly enlarge the class centers.**

![](http://latex.codecogs.com/png.latex?\mathcal{L}=w_{intra}*\mathcal{L}_{intra}+w_{inter}*\mathcal{L}_{inter})

![](http://latex.codecogs.com/png.latex?\mathcal{L}_{intra}=\frac{\sum_{i=1}^{m}\exp(\alpha*\left\|\boldsymbol{x}_{i}-\boldsymbol{c}_{y_{i}}\right\|_{2}^{2})}{m})

![](http://latex.codecogs.com/png.latex?\mathcal{L}_{inter}=-\frac{\sum_{i=1}^{n_c}\sum_{j=1}^{n_c}I(i,j)*\log(\left\|\boldsymbol{c}_{i}-\boldsymbol{c}_{j}\right\|_{2}^{2})}{n_c})

where ![](http://latex.codecogs.com/png.latex?\alpha) is the scale factor, which is used for scaling the intra distance before applying the exponential function, this is required for numerical stabilization, ![](http://latex.codecogs.com/png.latex?n_c) is the number of classes,![](http://latex.codecogs.com/png.latex?I(i,j)=0)if ![](http://latex.codecogs.com/png.latex?i=j) otherwise ![](http://latex.codecogs.com/png.latex?I(i,j)=1), ![](http://latex.codecogs.com/png.latex?w_{intra}) and ![](http://latex.codecogs.com/png.latex?w_{inter}) are hyper parameters to balance the two supervision loss. This loss term has the similar effect as the one used in this repo on the mnist dataset, but failed to learn a set of good represents on fashion-mnist dataset and cifar10 dataset.

2. **build an extra fully connected layer over the class centers and apply softmax loss.**

This way, at least I can reduce the computation cost (batch_size vs num_classes), the final result is worse than the original center loss with softmax loss, probably it is because the centers are already separable but indeed they are closed to each other.

## Rethinking Softmax Loss

Does softmax loss only focus on learning separable features? What is the relationship between increasing the inter class separability and increasing the inter class distance? Will softmax loss stop to increase inter class distance when the learned features are already well separated? Would it possible for the features learned by softmax loss to reflect the relationship across the dataset?

## Acknowledgement

Thanks Google for providing free access to their Colab service, without this service I could not finish all the experiments in one month.

