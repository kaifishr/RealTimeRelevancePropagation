"""Python class that implements the layer-wise relevance propagation algorithm.

   Typical usage example:

    lrp = RelevancePropagation(conf)
    lrp.run(image)
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import gen_nn_ops
from tensorflow.keras.applications.vgg16 import preprocess_input

#tf.compat.v1.disable_eager_execution()  # uncomment if TypeError occurs


class RelevancePropagation(object):
    def __init__(self, conf):
        input_shape = (conf["image"]["height"], conf["image"]["width"], conf["image"]["channels"])
        network = conf["model"]["name"]
        weights = conf["model"]["weights"]
        self.epsilon = conf["lrp"]["epsilon"]
        self.rule = conf["lrp"]["rule"]
        self.grayscale = conf["lrp"]["grayscale"]

        # Load model
        if network == "vgg16":
            self.model = tf.keras.applications.VGG16(input_shape=input_shape, weights=weights)
        elif network == "vgg19":
            self.model = tf.keras.applications.VGG19(input_shape=input_shape, weights=weights)
        else:
            raise Exception("Error: Unknown network name.")

        # Extract model's weights
        self.weights = {weight.name.split('/')[0]: weight for weight in self.model.trainable_weights if 'bias' not in weight.name}

        # Extract activation layers
        self.activations = [layer.output for layer in self.model.layers]
        self.activations = self.activations[::-1]

        # --- Extract the model's layers name
        self.layer_names = [layer.name for layer in self.model.layers if 'dropout' not in layer.name]
        self.layer_names = self.layer_names[::-1]

        # --- Build relevance graph ---
        self.relevance = self.relevance_propagation()

        self.f = K.function(inputs=self.model.input, outputs=self.relevance)

    def run(self, image):
        """Computes feature relevance scores for single image

        :param image: ndarray of shape (W, H, C)
        :return: features_relevance_scores: ndarray of same size as input image
        """
        image = preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        relevance_scores = self.f(inputs=image)
        relevance_scores = self.postprocess(relevance_scores)
        return np.squeeze(relevance_scores)

    def relevance_propagation(self):
        relevance = self.model.output
        for i, layer_name in enumerate(self.layer_names):
            if 'prediction' in layer_name:
                relevance = self.relprop_dense(self.activations[i+1], self.weights[layer_name], relevance)
            elif 'fc' in layer_name:
                relevance = self.relprop_dense(self.activations[i+1], self.weights[layer_name], relevance)
            elif 'flatten' in layer_name:
                relevance = self.relprop_flatten(self.activations[i+1], relevance)
            elif 'pool' in layer_name:
                relevance = self.relprop_pool(self.activations[i+1], relevance)
            elif 'conv' in layer_name:
                relevance = self.relprop_conv(self.activations[i+1], self.weights[layer_name], relevance, layer_name)
            elif 'input' in layer_name:
                pass
            else:
                raise Exception('Layer type not recognized.')
        return relevance

    def relprop_dense(self, a, w, r):
        w_pos = tf.maximum(w, 0.0)
        z = tf.matmul(a, w_pos) + self.epsilon
        s = r / z
        c = tf.matmul(s, tf.transpose(w_pos))
        return c * a

    def relprop_flatten(self, a, r):
        return tf.reshape(r, tf.shape(a))

    def relprop_pool(self, a, r, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', operation='avg'):
        if operation == 'avg':
            z = tf.nn.avg_pool(a, ksize, strides, padding) + self.epsilon
            s = r / z
            c = gen_nn_ops.avg_pool_grad(tf.shape(a), s, ksize, strides, padding)
        elif operation == 'max':
            z = tf.nn.max_pool(a, ksize, strides, padding) + self.epsilon
            s = r / z
            c = gen_nn_ops.max_pool_grad_v2(a, z, s, ksize, strides, padding)
        else:
            raise Exception('No such unpooling operation.')
        return c * a

    def relprop_conv(self, a, w, r, name, strides=(1, 1, 1, 1), padding='SAME'):
        if name == 'block1_conv1':
            a = tf.ones_like(a)     # for the input
        w_pos = tf.maximum(w, 0.0)
        z = tf.nn.conv2d(a, w_pos, strides, padding) + self.epsilon
        s = r / z
        c = tf.compat.v1.nn.conv2d_backprop_input(tf.shape(a), w_pos, s, strides, padding)
        return c * a

    @staticmethod
    def rescale(x):
        """
        Rescales values of a batch of images between 0 and 1

        :param x: ndarray of dimensions (N, W, H, C)
        :return: rescaled images of same dimension as input
        """
        x_min = np.min(x, axis=(1, 2), keepdims=True)
        x_max = np.max(x, axis=(1, 2), keepdims=True)
        return (x - x_min).astype("float64") / (x_max - x_min).astype("float64")

    def postprocess(self, x):
        """Postprocesses batch of feature relevance scores (relevance_maps).

        Args:
            x: array with dimension (N, W, H, C)

        Returns:
            x: array with dimensions (N, W, H, C) or (N, W, H)

        """
        if self.grayscale:
            x = np.mean(x, axis=-1)
        x = self.rescale(x)
        return x
