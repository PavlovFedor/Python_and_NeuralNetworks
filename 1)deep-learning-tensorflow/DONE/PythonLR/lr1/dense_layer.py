import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lr_common')))
from activations import get_activation

#actvation = [linear, sigmoid, ReLU, softplus, mish]
class DenseLayer(Layer):
    def __init__(self, units, activation='linear', **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        
    def build(self, input_shape):
        # Создаём веса (W) размером [input_dim, units]
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        # Создаём смещения (b) размером [units]
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        
    def call(self, inputs):
        # Линейная комбинация: inputs @ w + b
        output = tf.matmul(inputs, self.w) + self.b
        # Применяем функцию активации
        return self._apply_activation(output)
    
    def _apply_activation(self, x):
        act_fn = get_activation(self.activation_name)
        return act_fn(x)