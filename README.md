# RealTimeRelevancePropagation

*Update: An improved version of real-time relevance propagation for PyTorch can be found [here](https://github.com/kaifishr/PyTorchRelevancePropagation).*

An implementation of real time layer-wise relevance propagation (LRP) with a pre-trained VGG16 network using OpenCV and Tensorflow. 

The example below shows the result of a short test. It is clearly visible that the relevance values in the background, especially at the door, disappear entirely when the notebook enters the image area, which is then assigned a lot of relevance. This is to be expected since the VGG16 network has also been trained to classify notebooks.

<p align="center">
    <img src="./video/test.gif" height="224">
</p>

A fast graphics card is recommended for a better experience. With an RTX 2080 Ti, I get around 33 FPS.
