import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lr_common')))
from activations import get_activation

class CustomLSTM(layers.Layer):
    """
    Кастомный LSTM слой с поддержкой:
    - количества единиц памяти (units)
    - return_sequences (True/False)
    - activation (функция активации для клеточной активации, по умолчанию tanh)
    - recurrent_activation (функция активации для ворот, по умолчанию sigmoid)
    """
    
    def __init__(self,
                 units,
                 return_sequences=False,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.units = units
        self.return_sequences = return_sequences
        self.activation = activation.lower()
        self.recurrent_activation = recurrent_activation.lower()
        self.use_bias = use_bias
        
        # Валидация параметров
        valid_activations = ['tanh', 'relu', 'sigmoid']
        if self.activation not in valid_activations:
            raise ValueError(f"Activation must be one of {valid_activations}, got {activation}")
        
        valid_recurrent_activations = ['sigmoid', 'hard_sigmoid', 'tanh']
        if self.recurrent_activation not in valid_recurrent_activations:
            raise ValueError(f"Recurrent activation must be one of {valid_recurrent_activations}, got {recurrent_activation}")
    
    def build(self, input_shape):
        """Создание обучаемых весов"""
        # input_shape: (batch, timesteps, features)
        self.input_dim = input_shape[-1]
        
        # Веса для входных связей (W)
        self.w_i = self.add_weight(
            name='w_i',
            shape=(self.input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        self.w_f = self.add_weight(
            name='w_f',
            shape=(self.input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        self.w_c = self.add_weight(
            name='w_c',
            shape=(self.input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        self.w_o = self.add_weight(
            name='w_o',
            shape=(self.input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        
        # Веса для рекуррентных связей (U)
        self.u_i = self.add_weight(
            name='u_i',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        self.u_f = self.add_weight(
            name='u_f',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        self.u_c = self.add_weight(
            name='u_c',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        self.u_o = self.add_weight(
            name='u_o',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        
        # Смещения (bias)
        if self.use_bias:
            self.b_i = self.add_weight(
                name='b_i',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
                dtype=tf.float32
            )
            self.b_f = self.add_weight(
                name='b_f',
                shape=(self.units,),
                initializer='ones',  # bias забывающих ворот инициализируем единицами
                trainable=True,
                dtype=tf.float32
            )
            self.b_c = self.add_weight(
                name='b_c',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
                dtype=tf.float32
            )
            self.b_o = self.add_weight(
                name='b_o',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
                dtype=tf.float32
            )
        
        super().build(input_shape)
    
    def _get_activation(self, x, activation_type):
        """Вспомогательный метод для выбора функции активации"""
        act_fn = get_activation(activation_type)
        return act_fn(x)
    
    def _step(self, inputs, states):
        """
        Один шаг LSTM
        inputs: вход на текущем шаге (batch, features)
        states: (h_tm1, c_tm1) - состояния с предыдущего шага
        """
        h_tm1, c_tm1 = states
        
        # Входные ворота (input gate)
        i_t = self._get_activation(
            tf.matmul(inputs, self.w_i) + tf.matmul(h_tm1, self.u_i) + (self.b_i if self.use_bias else 0),
            self.recurrent_activation
        )
        
        # Забывающие ворота (forget gate)
        f_t = self._get_activation(
            tf.matmul(inputs, self.w_f) + tf.matmul(h_tm1, self.u_f) + (self.b_f if self.use_bias else 0),
            self.recurrent_activation
        )
        
        # Клеточные ворота (cell gate)
        c_tilde = self._get_activation(
            tf.matmul(inputs, self.w_c) + tf.matmul(h_tm1, self.u_c) + (self.b_c if self.use_bias else 0),
            self.activation
        )
        
        # Обновление состояния ячейки
        c_t = f_t * c_tm1 + i_t * c_tilde
        
        # Выходные ворота (output gate)
        o_t = self._get_activation(
            tf.matmul(inputs, self.w_o) + tf.matmul(h_tm1, self.u_o) + (self.b_o if self.use_bias else 0),
            self.recurrent_activation
        )
        
        # Скрытое состояние
        h_t = o_t * self._get_activation(c_t, self.activation)
        
        return h_t, c_t
    
    def call(self, inputs, states=None, **kwargs):
        """
        Прямой проход LSTM
        inputs: (batch, timesteps, features)
        states: (h0, c0) - начальные состояния (опционально)
        """
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        
        # Инициализация состояний нулями
        if states is not None:
            h_t = states[0]
            c_t = states[1]
        else:
            h_t = tf.zeros((batch_size, self.units), dtype=tf.float32)
            c_t = tf.zeros((batch_size, self.units), dtype=tf.float32)
        
        # Используем tf.while_loop вместо Python for
        time = tf.constant(0, dtype=tf.int32)
        
        def cond(time, h_t, c_t, outputs):
            return time < timesteps
        
        def body(time, h_t, c_t, outputs):
            x_t = inputs[:, time, :]
            h_t, c_t = self._step(x_t, (h_t, c_t))
            
            # Обновляем outputs в зависимости от return_sequences
            if self.return_sequences:
                outputs = outputs.write(time, h_t)
            
            return time + 1, h_t, c_t, outputs
        
        # Создаем TensorArray для сбора выходов
        outputs = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps if self.return_sequences else 0,
            dynamic_size=False
        )
        
        # Запускаем цикл
        _, h_final, c_final, outputs = tf.while_loop(
            cond, body, [time, h_t, c_t, outputs],
            parallel_iterations=10,
            swap_memory=True
        )
        
        # Формирование выхода
        if self.return_sequences:
            output = outputs.stack()  # (timesteps, batch, units)
            output = tf.transpose(output, [1, 0, 2])  # (batch, timesteps, units)
        else:
            output = h_final  # (batch, units)
        
        return output
    
    def compute_output_shape(self, input_shape):
        """Вычисление формы выходного тензора"""
        batch, timesteps, features = input_shape
        
        if self.return_sequences:
            return (batch, timesteps, self.units)
        else:
            return (batch, self.units)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
            'use_bias': self.use_bias
        })
        return config