import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from custom_conv import CustomConv2D


def preprocess_cifar10():
    """Загрузка и предобработка данных CIFAR-10"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Нормализация
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


def preprocess_mnist():
    """Загрузка и предобработка данных MNIST"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Нормализация и добавление канального измерения
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # One-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


def create_model_cifar10():
    """create_model_cifar10
    Создание модели для CIFAR-10 с тремя кастомными сверточными слоями
    """
    model = models.Sequential()
    
    # Первый сверточный блок (ReLU активация)
    model.add(CustomConv2D(
        out_channels=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
        input_shape=(32, 32, 3)
    ))
    model.add(layers.BatchNormalization())
    model.add(CustomConv2D(
        out_channels=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.25))
    
    # Второй сверточный блок (LeakyReLU активация)
    model.add(CustomConv2D(
        out_channels=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='leaky_relu'
    ))
    model.add(layers.BatchNormalization())
    model.add(CustomConv2D(
        out_channels=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='leaky_relu'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.25))
    
    # Третий сверточный блок (Softplus активация)
    model.add(CustomConv2D(
        out_channels=128,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='softplus'
    ))
    model.add(layers.BatchNormalization())
    model.add(CustomConv2D(
        out_channels=128,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='softplus'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.25))
    
    # Полносвязная часть
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model


def create_model_mnist():
    """
    Создание модели для MNIST с тремя кастомными сверточными слоями
    """
    model = models.Sequential()
    
    # Первый сверточный блок (Sigmoid активация)
    model.add(CustomConv2D(
        out_channels=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
        input_shape=(28, 28, 1)
    ))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    
    # Второй сверточный блок (Tanh активация)
    model.add(CustomConv2D(
        out_channels=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    
    # Третий сверточный блок (ReLU активация)
    model.add(CustomConv2D(
        out_channels=128,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu'
    ))
    model.add(layers.GlobalAveragePooling2D())
    
    # Полносвязная часть
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model


def create_model_with_different_activations():
    """
    Модель для демонстрации различных функций активации в одном потоке
    """
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Слой с linear активацией (без активации)
    x = CustomConv2D(32, 3, activation='linear', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Слой с ReLU активацией
    x = CustomConv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Слой с Sigmoid активацией
    x = CustomConv2D(128, 3, activation='sigmoid', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Слой с Softplus активацией
    x = CustomConv2D(256, 3, activation='softplus', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    return models.Model(inputs, outputs)


def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=10, model_name="Model"):
    """Обучение и оценка модели"""
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    
    print(f"\n{'='*50}")
    print(f"Обучение модели: {model_name}")
    print(f"{'='*50}")
    
    # Обучение
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Оценка
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{model_name} - Точность на тесте: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{model_name} - Ошибка на тесте: {test_loss:.4f}")
    
    return history, test_acc, test_loss


def plot_training_history(history, model_name):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # График точности
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # График потерь
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'lr3/training_history_{model_name.replace(" ", "_")}.png', dpi=150)
    plt.show()


def test_custom_conv_layer():
    """Тестирование кастомного сверточного слоя"""
    print("\n" + "="*60)
    print("ЧАСТЬ 1: ТЕСТИРОВАНИЕ КАСТОМНОГО СВЕРТОЧНОГО СЛОЯ")
    print("="*60)
    
    print("\n1.1 Проверка работы слоя на случайных данных:")
    test_input = tf.random.normal([2, 32, 32, 3])
    
    # Тест различных конфигураций
    configs = [
        {"out_channels": 16, "kernel_size": 3, "strides": 1, "padding": "valid", "activation": "relu"},
        {"out_channels": 32, "kernel_size": 5, "strides": 2, "padding": "same", "activation": "sigmoid"},
        {"out_channels": 64, "kernel_size": 3, "strides": 1, "padding": "valid", "activation": "tanh"},
        {"out_channels": 128, "kernel_size": 1, "strides": 1, "padding": "same", "activation": "softplus"},
        {"out_channels": 32, "kernel_size": 3, "strides": 2, "padding": "valid", "activation": "leaky_relu"},
    ]
    
    for i, config in enumerate(configs, 1):
        layer = CustomConv2D(**config)
        output = layer(test_input)
        print(f"  Тест {i}: {config['out_channels']} фильтров, "
              f"kernel={config['kernel_size']}, stride={config['strides']}, "
              f"padding={config['padding']}, activation={config['activation']}")
        print(f"       Вход: {test_input.shape} -> Выход: {output.shape}")
    
    # Проверка сохранения/загрузки конфигурации
    layer = CustomConv2D(32, 3, activation='relu')
    config = layer.get_config()
    new_layer = CustomConv2D.from_config(config)
    print(f"\n1.2 Сохранение/загрузка конфигурации слоя: УСПЕШНО")


if __name__ == "__main__":
    # Устанавливаем random seed для воспроизводимости
    tf.random.set_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА №3")
    print("Кастомный сверточный слой для TensorFlow")
    print("="*60)
    
    # ========== ЧАСТЬ 1: ТЕСТИРОВАНИЕ КАСТОМНОГО СЛОЯ ==========
    test_custom_conv_layer()
    
    # ========== ЧАСТЬ 2: ОБУЧЕНИЕ НА MNIST ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 2: ОБУЧЕНИЕ НА MNIST")
    print("="*60)
    
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = preprocess_mnist()
    
    # Используем часть данных для ускорения
    x_train_mnist_small = x_train_mnist[:10000]
    y_train_mnist_small = y_train_mnist[:10000]
    x_test_mnist_small = x_test_mnist[:2000]
    y_test_mnist_small = y_test_mnist[:2000]
    
    print(f"\nРазмер обучающей выборки MNIST: {x_train_mnist_small.shape}")
    print(f"Размер тестовой выборки MNIST: {x_test_mnist_small.shape}")
    
    model_mnist = create_model_mnist()
    model_mnist.summary()
    
    history_mnist, acc_mnist, loss_mnist = train_and_evaluate(
        model_mnist,
        x_train_mnist_small, y_train_mnist_small,
        x_test_mnist_small, y_test_mnist_small,
        epochs=5,
        model_name="MNIST CNN с кастомными слоями"
    )
    
    plot_training_history(history_mnist, "MNIST")
    
    # ========== ЧАСТЬ 3: ОБУЧЕНИЕ НА CIFAR-10 ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 3: ОБУЧЕНИЕ НА CIFAR-10")
    print("="*60)
    
    (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = preprocess_cifar10()
    
    # Используем часть данных для ускорения
    x_train_cifar_small = x_train_cifar[:5000]
    y_train_cifar_small = y_train_cifar[:5000]
    x_test_cifar_small = x_test_cifar[:1000]
    y_test_cifar_small = y_test_cifar[:1000]
    
    print(f"\nРазмер обучающей выборки CIFAR-10: {x_train_cifar_small.shape}")
    print(f"Размер тестовой выборки CIFAR-10: {x_test_cifar_small.shape}")
    
    model_cifar = create_model_cifar10()
    model_cifar.summary()
    
    history_cifar, acc_cifar, loss_cifar = train_and_evaluate(
        model_cifar,
        x_train_cifar_small, y_train_cifar_small,
        x_test_cifar_small, y_test_cifar_small,
        epochs=5,
        model_name="CIFAR-10 CNN с кастомными слоями"
    )
    
    plot_training_history(history_cifar, "CIFAR10")
    
    # ========== ЧАСТЬ 4: МОДЕЛЬ С РАЗЛИЧНЫМИ АКТИВАЦИЯМИ ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 4: МОДЕЛЬ С РАЗЛИЧНЫМИ ФУНКЦИЯМИ АКТИВАЦИИ")
    print("="*60)
    
    model_activations = create_model_with_different_activations()
    model_activations.summary()
    
    history_activations, acc_activations, loss_activations = train_and_evaluate(
        model_activations,
        x_train_cifar_small, y_train_cifar_small,
        x_test_cifar_small, y_test_cifar_small,
        epochs=3,
        model_name="Модель с разными активациями"
    )
    
    # ========== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ==========
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЛАБОРАТОРНОЙ РАБОТЫ №3")
    print("="*60)
    
    print("\nРезультаты обучения:")
    print("-" * 50)
    print(f"  MNIST (кастомные сверточные слои):")
    print(f"    - Точность: {acc_mnist:.4f} ({acc_mnist*100:.2f}%)")
    print(f"    - Ошибка: {loss_mnist:.4f}")
    print(f"\n  CIFAR-10 (кастомные сверточные слои):")
    print(f"    - Точность: {acc_cifar:.4f} ({acc_cifar*100:.2f}%)")
    print(f"    - Ошибка: {loss_cifar:.4f}")
    print(f"\n  CIFAR-10 (разные функции активации):")
    print(f"    - Точность: {acc_activations:.4f} ({acc_activations*100:.2f}%)")
    print(f"    - Ошибка: {loss_activations:.4f}")
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("1. Кастомный сверточный слой успешно реализован и совместим с TensorFlow")
    print("2. Слой поддерживает все требуемые параметры:")
    print("   - Количество фильтров (out_channels)")
    print("   - Размер ядра (kernel_size) и шаг (strides)")
    print("   - Тип паддинга (SAME/VALID)")
    print("   - Функции активации (linear, relu, sigmoid, tanh, softplus, leaky_relu)")
    print("3. Построена нейронная сеть с тремя кастомными сверточными слоями")
    print("4. Модель успешно обучена на наборах данных MNIST и CIFAR-10")
    print("5. Использованы оптимизатор Adam и функция потерь categorical_crossentropy")
    print("="*60)