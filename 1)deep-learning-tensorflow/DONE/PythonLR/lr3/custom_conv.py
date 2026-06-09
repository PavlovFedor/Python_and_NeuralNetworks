import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lr_common')))
from activations import get_activation

class CustomConv2D(layers.Layer):
    """
    Кастомный сверточный слой с поддержкой:
    - количества фильтров (out_channels)
    - размера ядра (kernel_size)
    - шага (stride)
    - типа паддинга (SAME/VALID)
    - функции активации (linear, relu, sigmoid, tanh, softplus, leaky_relu)
    """
    
    def __init__(self,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 activation='linear',
                 use_bias=True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding = padding.upper()
        self.activation = activation.lower()
        self.use_bias = use_bias
        
        # Валидация параметров
        if self.padding not in ['SAME', 'VALID']:
            raise ValueError(f"Padding must be 'SAME' or 'VALID', got {padding}")
        
        valid_activations = ['linear', 'relu', 'sigmoid', 'tanh', 'softplus', 'leaky_relu']
        if self.activation not in valid_activations:
            raise ValueError(f"Activation must be one of {valid_activations}, got {activation}")
    
    def build(self, input_shape):
        """Создание обучаемых весов"""
        self.in_channels = input_shape[-1]
        
        # Ядро свертки (фильтры)
        kernel_shape = (*self.kernel_size, self.in_channels, self.out_channels)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        
        # Смещение (bias)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.out_channels,),
                initializer='zeros',
                trainable=True,
                dtype=tf.float32
            )
        
        super().build(input_shape)
    
    def _get_activation(self, x):
        """Вспомогательный метод для выбора функции активации"""
        act_fn = get_activation(self.activation)
        return act_fn(x)
    
    def call(self, inputs):
        """Прямой проход: свертка + смещение + активация"""
        # Ручная реализация свертки
        output = self._conv2d_manual(inputs, self.kernel, self.strides, self.padding)
        
        # Добавление смещения
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        
        # Применение функции активации
        output = self._get_activation(output)
        
        return output
    
    def compute_output_shape(self, input_shape):
        """Вычисление формы выходного тензора"""
        batch, height, width, channels = input_shape
        
        if self.padding == 'SAME':
            out_h = int(np.ceil(height / self.strides[0]))
            out_w = int(np.ceil(width / self.strides[1]))
        else:  # VALID
            out_h = int(np.ceil((height - self.kernel_size[0] + 1) / self.strides[0]))
            out_w = int(np.ceil((width - self.kernel_size[1] + 1) / self.strides[1]))
        
        return (batch, out_h, out_w, self.out_channels)
    
    def _conv2d_manual(self, inputs, kernel, strides, padding):
        """ Ручная реализация свертки через извлечение патчей """ 
        batch_size = tf.shape(inputs)[0]
        in_h = tf.shape(inputs)[1]
        in_w = tf.shape(inputs)[2]
        in_c = tf.shape(inputs)[3]
        
        k_h, k_w = self.kernel_size
        s_h, s_w = strides
        out_c = self.out_channels
        
        # Вычисление выходных размеров и padding
        if padding == 'SAME':
            out_h = tf.cast(tf.math.ceil(tf.cast(in_h, tf.float32) / s_h), tf.int32)
            out_w = tf.cast(tf.math.ceil(tf.cast(in_w, tf.float32) / s_w), tf.int32)
            
            # Ручной padding нулями
            pad_h = tf.maximum(0, (out_h - 1) * s_h + k_h - in_h)
            pad_w = tf.maximum(0, (out_w - 1) * s_w + k_w - in_w)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            inputs = tf.pad(inputs, [
                [0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]
            ])
            
            # Обновляем размеры после padding
            in_h = tf.shape(inputs)[1]
            in_w = tf.shape(inputs)[2]
        else:  # VALID
            out_h = tf.cast(tf.math.ceil(tf.cast(in_h - k_h + 1, tf.float32) / s_h), tf.int32)
            out_w = tf.cast(tf.math.ceil(tf.cast(in_w - k_w + 1, tf.float32) / s_w), tf.int32)
        
        # Извлекаем патчи (окна)
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, k_h, k_w, 1],
            strides=[1, s_h, s_w, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        # Разворачиваем ядро в [k_h*k_w*in_c, out_c]
        # Применяет фильтр
        kernel_flat = tf.reshape(kernel, [k_h * k_w * in_c, out_c])
        
        # Свёртка через матричное умножение
        output = tf.matmul(patches, kernel_flat)
        
        #kernel_height, kernel_width - размер окна(h x w; 3 x 3)
        #in_channels - каналы RGB
        #out_channels - колво фильтров        
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'use_bias': self.use_bias
        })
        return config