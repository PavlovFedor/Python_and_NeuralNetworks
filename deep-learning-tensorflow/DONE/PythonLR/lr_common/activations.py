"""
Общий модуль с ручными реализациями функций активации
Для лабораторных работ №1, №2, №3, №4

Поддерживаемые функции активации:
- linear (без активации)
- relu (ReLU)
- sigmoid (Сигмоида)
- tanh (Гиперболический тангенс)
- softplus (Softplus)
- leaky_relu (Leaky ReLU)
- hard_sigmoid (Жёсткая сигмоида)
- mish (Mish)
- swish (Swish / SiLU)
- elu (Exponential Linear Unit)
- gelu (Gaussian Error Linear Unit)
- selu (Scaled Exponential Linear Unit)
"""

import tensorflow as tf


# ========== ОСНОВНЫЕ ФУНКЦИИ АКТИВАЦИИ ==========

def linear(x):
    """Линейная активация (без изменений)"""
    return x


def relu(x):
    """ReLU: max(0, x)"""
    return tf.maximum(0.0, x)


def sigmoid(x):
    """Сигмоида: 1 / (1 + exp(-x))"""
    return 1 / (1 + tf.exp(-x))


def tanh(x):
    """Гиперболический тангенс: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))


def softplus(x):
    """Softplus: log(1 + exp(x))"""
    return tf.math.log(1 + tf.exp(x))


def leaky_relu(x, alpha=0.1):
    """Leaky ReLU: x if x > 0 else alpha * x"""
    return tf.where(x > 0, x, alpha * x)


def hard_sigmoid(x):
    """
    Жёсткая сигмоида (аппроксимация сигмоиды для ускорения)
    clip((x + 1) / 2, 0, 1)
    """
    x = (x + 1) / 2
    return tf.clip_by_value(x, 0, 1)


def mish(x):
    """
    Mish: x * tanh(softplus(x))
    Более гладкая альтернатива ReLU
    """
    return x * tanh(softplus(x))


def swish(x, beta=1.0):
    """
    Swish (SiLU): x * sigmoid(beta * x)
    При beta = 1 совпадает с SiLU
    """
    return x * sigmoid(beta * x)


def elu(x, alpha=1.0):
    """
    ELU (Exponential Linear Unit)
    x if x > 0 else alpha * (exp(x) - 1)
    """
    return tf.where(x > 0, x, alpha * (tf.exp(x) - 1))


def gelu(x, approximate=True):
    """
    GELU (Gaussian Error Linear Unit)
    x * P(X <= x) where X ~ N(0,1)
    """
    if approximate:
        # Быстрая аппроксимация: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x**3)))
    else:
        # Точная формула через функцию ошибок
        return x * (0.5 * (1 + tf.math.erf(x / tf.sqrt(2.0))))


def selu(x):
    """
    SELU (Scaled Exponential Linear Unit)
    lambda * (x if x > 0 else alpha * (exp(x) - 1))
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * tf.where(x > 0, x, alpha * (tf.exp(x) - 1))


# ========== СЛОВАРЬ ДОСТУПНЫХ АКТИВАЦИЙ ==========

_ACTIVATIONS = {
    # Основные
    'linear': linear,
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'softplus': softplus,
    'leaky_relu': leaky_relu,
    'hard_sigmoid': hard_sigmoid,
    # Современные
    'mish': mish,
    'swish': swish,
    'silu': lambda x: swish(x, 1.0),  # SiLU = Swish с beta=1
    'elu': elu,
    'gelu': gelu,
    'selu': selu,
}


def get_activation(name, **kwargs):
    """
    Фабрика функций активации
    
    Args:
        name: Название функции активации (str)
        **kwargs: Дополнительные параметры (alpha для leaky_relu/elu, beta для swish)
    
    Returns:
        callable: Функция активации
    
    Examples:
        >>> act = get_activation('relu')
        >>> act = get_activation('leaky_relu', alpha=0.2)
        >>> act = get_activation('swish', beta=1.5)
    """
    name = name.lower()
    
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name}. "
            f"Available: {list(_ACTIVATIONS.keys())}"
        )
    
    # Обработка параметризованных активаций
    if name == 'leaky_relu':
        alpha = kwargs.get('alpha', 0.1)
        return lambda x: leaky_relu(x, alpha)
    elif name == 'swish' or name == 'silu':
        beta = kwargs.get('beta', 1.0)
        return lambda x: swish(x, beta)
    elif name == 'elu':
        alpha = kwargs.get('alpha', 1.0)
        return lambda x: elu(x, alpha)
    elif name == 'gelu':
        approximate = kwargs.get('approximate', True)
        return lambda x: gelu(x, approximate)
    else:
        return _ACTIVATIONS[name]


def get_available_activations():
    """Возвращает список всех доступных функций активации"""
    return list(_ACTIVATIONS.keys())


# ========== ДЛЯ ПРЯМОГО ИСПОЛЬЗОВАНИЯ В LAYER ==========

class ActivationWrapper:
    """
    Обёртка для использования функций активации в слоях
    Позволяет вызывать как act(x)
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.fn = get_activation(name, **kwargs)
    
    def __call__(self, x):
        return self.fn(x)
    
    def get_config(self):
        return {'name': self.name, **self.kwargs}
