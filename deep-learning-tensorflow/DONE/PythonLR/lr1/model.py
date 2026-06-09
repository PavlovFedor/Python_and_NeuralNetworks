import tensorflow as tf
from dense_layer import DenseLayer

def build_model():
    model = tf.keras.Sequential([ # type: ignore
        DenseLayer(64, activation='relu', input_shape=(20,)),  # 20 признаков (по умолчанию для make_classification)
        DenseLayer(32, activation='mish'),
        DenseLayer(1, activation='sigmoid')  # 1 выход для бинарной классификации
    ])
    return model