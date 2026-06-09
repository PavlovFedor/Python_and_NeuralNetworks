import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lr_common')))

class CustomPooling(layers.Layer):
    """
    Кастомный слой пулинга с ручной реализацией агрегации.
    Для извлечения окон используется tf.image.extract_patches (низкоуровневая операция),
    но все типы агрегации (max, average, median) реализованы вручную.
    """
    
    def __init__(self, 
                 pool_size=2, 
                 strides=2, 
                 padding='valid', 
                 aggregation='max',
                 pool_type='channelwise',
                 **kwargs):
        super().__init__(**kwargs)
        
        self.pool_size = pool_size if isinstance(pool_size, (tuple, list)) else (pool_size, pool_size)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding = padding.upper()
        self.aggregation = aggregation.lower()
        self.pool_type = pool_type.lower()
        
        # Валидация
        if self.padding not in ['SAME', 'VALID']:
            raise ValueError(f"Padding must be 'SAME' or 'VALID', got {padding}")
        if self.aggregation not in ['max', 'average', 'median']:
            raise ValueError(f"Aggregation must be 'max', 'average' or 'median', got {aggregation}")
        if self.pool_type not in ['channelwise', 'global']:
            raise ValueError(f"Pool type must be 'channelwise' or 'global', got {pool_type}")
    
    def call(self, inputs):
        if self.pool_type == 'global':
            return self._global_pooling_manual(inputs)
        else:
            return self._channelwise_pooling_manual(inputs)

    def _global_pooling_manual(self, inputs):
        """Ручная реализация глобального пулинга"""
        if self.aggregation == 'max':
            # максимум - последовательно по осям
            output = inputs
            for axis in [1, 2]:
                output = tf.reduce_max(output, axis=axis, keepdims=True)
        elif self.aggregation == 'average':
            # среднее
            height = tf.cast(tf.shape(inputs)[1], tf.float32)
            width = tf.cast(tf.shape(inputs)[2], tf.float32)
            total_pixels = height * width
            output = tf.reduce_sum(inputs, axis=[1, 2], keepdims=True)
            output = output / total_pixels
        elif self.aggregation == 'median':
            # медиана через сортировку
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            channels = tf.shape(inputs)[3]
            
            flattened = tf.reshape(inputs, [batch_size, height * width, channels])
            sorted_values = tf.sort(flattened, axis=1)
            mid = (height * width) // 2
            output = sorted_values[:, mid:mid+1, :]
            output = tf.reshape(output, [batch_size, 1, 1, channels])
        
        return output

    def _channelwise_pooling_manual(self, inputs):
        """
        Канальный пулинг с использованием extract_patches для извлечения окон,
        но с ручной реализацией агрегации.
        """
        # Получаем размеры патча
        patch_height = self.pool_size[0]
        patch_width = self.pool_size[1]
        
        # Извлекаем патчи
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, patch_height, patch_width, 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )
        
        # Получаем размеры
        batch_size = tf.shape(patches)[0]
        out_h = tf.shape(patches)[1]
        out_w = tf.shape(patches)[2]
        channels = tf.shape(inputs)[-1]
        patch_area = patch_height * patch_width
        
        # Преобразуем патчи в форму [batch, out_h, out_w, patch_area, channels]
        patches = tf.reshape(patches, [batch_size, out_h, out_w, patch_area, channels])
        
        # Ручная реализация агрегации
        if self.aggregation == 'max':
            output = tf.reduce_max(patches, axis=3)
                
        elif self.aggregation == 'average':
            # Ручное среднее
            output = tf.reduce_sum(patches, axis=3) / tf.cast(patch_area, tf.float32)
            
        elif self.aggregation == 'median':
            # Ручная медиана через сортировку
            # Транспонируем для сортировки по оси патчей
            patches_sorted = tf.sort(patches, axis=3)
            mid = patch_area // 2
            output = patches_sorted[:, :, :, mid, :]
        
        return output
    
    def compute_output_shape(self, input_shape):
        if self.pool_type == 'global':
            return (input_shape[0], 1, 1, input_shape[3])
        else:
            batch, height, width, channels = input_shape
            
            if self.padding == 'SAME':
                out_h = int(np.ceil(height / self.strides[0]))
                out_w = int(np.ceil(width / self.strides[1]))
            else:  # VALID
                out_h = int(np.ceil((height - self.pool_size[0] + 1) / self.strides[0]))
                out_w = int(np.ceil((width - self.pool_size[1] + 1) / self.strides[1]))
            
            if out_h < 1:
                out_h = 1
            if out_w < 1:
                out_w = 1
            
            return (batch, out_h, out_w, channels)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
            'aggregation': self.aggregation,
            'pool_type': self.pool_type
        })
        return config