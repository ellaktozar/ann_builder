import numpy as np
from abc import ABCMeta, abstractmethod

class Layer(metaclass=ABCMeta):
    def __init__(self, output_shape):
        self.output_layer = None
        self.output_shape = output_shape
        self.Z = None

    @abstractmethod
    def _activation(self):
        pass
    
    @abstractmethod
    def _derivative(self):
        pass

    def _backprop(self):
        self.grad = self.output_layer.grad.dot(
            self.output_layer.W.T
        ) * self._derivative(self.Z)
        self.input_layer._backprop()

    def _forward(self):
        self.Z = self._activation(
                self.input_layer.Z.dot(self.W) + self.b
            )
        if self.output_layer is not None:
            self.output_layer._forward()

    def _update_weights(self, learning_rate):
        self.W += learning_rate * self.input_layer.Z.T.dot(self.grad)
        self.b += learning_rate * self.grad.sum(axis=0)
        self.input_layer._update_weights(learning_rate)

    def init_weights(self):
        self.W = np.random.randn(self.input_layer.output_shape, self.output_shape) * 0.01
        self.b = np.zeros(self.output_shape)
        if self.output_layer is not None:
            self.output_layer.init_weights()

class ReluLayer(Layer):
    def _activation(self, Z):
        return Z * (Z > 0)

    def _derivative(self, Z):
        return (Z > 0)

class SoftmaxLayer(Layer):
    def _activation(self, Z):
        exp_z = np.exp(Z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _derivative(self, T, Z):
        return (T - Z)

    def _backprop(self):
        self.grad = self._derivative(self.T, self.Z)
        self.input_layer._backprop()
    
    def _forward(self):
        self.Z = self._activation(
            self.input_layer.Z.dot(self.W) + self.b
        )

    def init_weights(self):
        self.W = np.random.randn(self.input_layer.output_shape, self.output_shape) * 0.01
        self.b = np.zeros(self.output_shape)

class InputLayer():
    def __init__(self):
        self.output_shape = None
        self.output_layer = None

    def _backprop(self):
        pass

    def _update_weights(self, learning_rate):
        pass
    
    def init_weights(self):
        self.output_layer.init_weights()

class SigmoidLayer(Layer):
    def _activation(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _derivative(self, Z):
        return self._activation(Z) * (1 - self._activation(Z))

class TanhLayer(Layer):
    def _activation(self, Z):
        return (np.exp(Z) - np.exp(-Z)) /\
        (np.exp(Z) + np.exp(-Z))

    def _derivative(self, Z):
        temp = self._activation(Z)
        return 1 - (temp ** 2)