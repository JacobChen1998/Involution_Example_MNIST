# Simple_Involution_Example_MNIST
This is a simple example that uses involution layer instead of convolution layer in MNIST classification task.

The involution network was perposed by [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/abs/2103.06255) (CVPR'21).
The original work can reach [their GitHub page](https://github.com/d-li14/involution/), which is a PyTorch-based.

Here we follow their concept and modify model of [keras MNIST example](https://keras.io/examples/vision/mnist_convnet/) with two following motions:
1. Replace the CONV2D layer with the involution layer.
2. Remove Maxpooling2D to raise the training parameters number.

|         Model         | Params(M) | Accuracy (20 Epochs) |
:---------------------:|:---------:|:--------:|
| Convolution-based    |  34,826 | 0.992  | 
| Involution-based    |  7,898 | 0.935  | 
| Involution-based (Res1)    |  7,898 | 0.939  | 
| Involution-based (Res2)    |  7,898 | 0.919  | 
| Involution-based (with Maxpooling2D)    |  548 | 0.866  |

![N](https://github.com/JacobChen1998/Involution_Example_MNIST/blob/main/Network.png)

Using the Involution layer to replace Convulution layer, the parameters number is only 63 times of same structure but CNN.
