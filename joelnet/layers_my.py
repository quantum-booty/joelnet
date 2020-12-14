"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward and propagate gradients backward.
For examplea neural net might look like
inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Dict, Callable
import numpy as np
from joelnet.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = dict()
        self.grads: Dict[str, Tensor] = dict()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """

        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    Computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad: Tensor) -> Tensor:
        """
        we differentiate loss wrt input
        i = input, o = output, b = bias, L = loss funcion
        if L = L(o) and o = i * w + b
        then dL/di = L'(o) * w
        and dL/dw = L'(o) * i
        and dL/db = L'(o)

        if L = L(o) and o = i @ w + b
        << then dL/di = L'(o) @ w.T >> This is what we need
        and dL/dw = i.T @ L'(o)
        and dL/db = L'(o)
        """
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    An activation layer just applies a function elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(x)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y**2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
