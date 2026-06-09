import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb, reuters
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from custom_lstm import CustomLSTM


def preprocess_imdb_data(max_features=10000, maxlen=100):
    """Загрузка и предобработка данных IMDB"""
    print("\nЗагрузка датасета IMDB...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    
    # Паддинг последовательностей
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    
    print(f"Размер обучающей выборки: {x_train.shape}")
    print(f"Размер тестовой выборки: {x_test.shape}")
    print(f"Классы: {np.unique(y_train)}")
    
    return (x_train, y_train), (x_test, y_test)


def preprocess_reuters_data(max_features=5000, maxlen=100):
    """Загрузка и предобработка данных Reuters"""
    print("\nЗагрузка датасета Reuters...")
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)
    
    # Паддинг последовательностей
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    
    # One-hot encoding для меток
    num_classes = max(np.max(y_train), np.max(y_test)) + 1
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"Размер обучающей выборки: {x_train.shape}")
    print(f"Размер тестовой выборки: {x_test.shape}")
    print(f"Количество классов: {num_classes}")
    
    return (x_train, y_train), (x_test, y_test), num_classes


def create_model_imdb(max_features=10000, maxlen=100, units1=64, units2=32):
    """
    Создание модели для IMDB с двумя кастомными LSTM слоями
    """
    model = models.Sequential()
    
    # Embedding слой для преобразования слов в векторы
    model.add(layers.Embedding(max_features, 128, input_length=maxlen))
    
    # Первый LSTM слой (возвращает последовательность)
    model.add(CustomLSTM(
        units=units1,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid'
    ))
    
    # Второй LSTM слой (возвращает только последний выход)
    model.add(CustomLSTM(
        units=units2,
        return_sequences=False,
        activation='tanh',
        recurrent_activation='sigmoid'
    ))
    
    # Dropout для регуляризации
    model.add(layers.Dropout(0.5))
    
    # Полносвязный слой
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Выходной слой (бинарная классификация)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model


def create_model_reuters(max_features=5000, maxlen=100, num_classes=46, units1=128, units2=64):
    """
    Создание модели для Reuters с двумя кастомными LSTM слоями
    """
    model = models.Sequential()
    
    # Embedding слой
    model.add(layers.Embedding(max_features, 128, input_length=maxlen))
    
    # Первый LSTM слой
    model.add(CustomLSTM(
        units=units1,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid'
    ))
    
    # Второй LSTM слой
    model.add(CustomLSTM(
        units=units2,
        return_sequences=False,
        activation='tanh',
        recurrent_activation='sigmoid'
    ))
    
    # Полносвязные слои
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Выходной слой (многоклассовая классификация)
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_deep_lstm_model(max_features=10000, maxlen=100):
    """
    Создание глубокой модели с тремя кастомными LSTM слоями
    """
    model = models.Sequential()
    
    model.add(layers.Embedding(max_features, 128, input_length=maxlen))
    
    # Три LSTM слоя с возвратом последовательностей
    model.add(CustomLSTM(64, return_sequences=True))
    model.add(CustomLSTM(64, return_sequences=True))
    model.add(CustomLSTM(32, return_sequences=False))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model


def train_and_evaluate(model, x_train, y_train, x_test, y_test, 
                       epochs=10, batch_size=64, model_name="Model", is_binary=True):
    """Обучение и оценка модели"""
    
    # Выбор функции потерь
    if is_binary:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
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
        batch_size=batch_size,
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
    
    # СОЗДАНИЕ ПАПКИ lr4 ЕСЛИ ОНА НЕ СУЩЕСТВУЕТ
    save_dir = 'lr4'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Создана папка: {save_dir}")
    
    # СОХРАНЕНИЕ В ПАПКУ lr4
    save_path = os.path.join(save_dir, f'training_history_{model_name.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=150)
    print(f"График сохранен: {save_path}")
    plt.show()


def test_custom_lstm_layer():
    """Тестирование кастомного LSTM слоя"""
    print("\n" + "="*60)
    print("ЧАСТЬ 1: ТЕСТИРОВАНИЕ КАСТОМНОГО LSTM СЛОЯ")
    print("="*60)
    
    print("\n1.1 Проверка работы слоя на случайных данных:")
    
    # Тест 1: return_sequences=False
    test_input = tf.random.normal([32, 50, 100])  # (batch, timesteps, features)
    lstm1 = CustomLSTM(units=64, return_sequences=False)
    output1 = lstm1(test_input)
    print(f"  return_sequences=False:")
    print(f"       Вход: {test_input.shape} -> Выход: {output1.shape}")
    
    # Тест 2: return_sequences=True
    lstm2 = CustomLSTM(units=64, return_sequences=True)
    output2 = lstm2(test_input)
    print(f"  return_sequences=True:")
    print(f"       Вход: {test_input.shape} -> Выход: {output2.shape}")
    
    # Тест 3: С различными активациями
    configs = [
        {"units": 32, "return_sequences": False, "activation": "tanh", "recurrent_activation": "sigmoid"},
        {"units": 64, "return_sequences": True, "activation": "relu", "recurrent_activation": "sigmoid"},
        {"units": 128, "return_sequences": False, "activation": "tanh", "recurrent_activation": "hard_sigmoid"},
    ]
    
    for i, config in enumerate(configs, 1):
        lstm = CustomLSTM(**config)
        output = lstm(test_input)
        print(f"  Тест {i}: units={config['units']}, activation={config['activation']}, "
              f"recurrent_activation={config['recurrent_activation']}")
        print(f"       Вход: {test_input.shape} -> Выход: {output.shape}")
    
    # Проверка с начальным состоянием
    print("\n1.2 Проверка с передачей начального состояния:")
    h0 = tf.random.normal([32, 64])
    c0 = tf.random.normal([32, 64])
    lstm3 = CustomLSTM(units=64, return_sequences=False)
    output3 = lstm3(test_input, initial_state=(h0, c0)) # type: ignore
    print(f"  Переданы начальные состояния h0={h0.shape}, c0={c0.shape}")
    print(f"  Выход: {output3.shape}")
    
    # Проверка сохранения/загрузки конфигурации
    lstm4 = CustomLSTM(128, return_sequences=True)
    config = lstm4.get_config()
    new_lstm = CustomLSTM.from_config(config)
    print(f"\n1.3 Сохранение/загрузка конфигурации слоя: УСПЕШНО")


if __name__ == "__main__":
    # Устанавливаем random seed для воспроизводимости
    tf.random.set_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА №4")
    print("Кастомный LSTM слой для TensorFlow")
    print("="*60)
    
    # ========== ЧАСТЬ 1: ТЕСТИРОВАНИЕ КАСТОМНОГО СЛОЯ ==========
    test_custom_lstm_layer()
    
    # ========== ЧАСТЬ 2: ОБУЧЕНИЕ НА IMDB ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 2: ОБУЧЕНИЕ НА IMDB (БИНАРНАЯ КЛАССИФИКАЦИЯ)")
    print("="*60)
    
    # Загрузка данных
    max_features = 10000
    maxlen = 100
    (x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = preprocess_imdb_data(max_features, maxlen)
    
    # Используем часть данных для ускорения
    x_train_imdb_small = x_train_imdb[:5000]
    y_train_imdb_small = y_train_imdb[:5000]
    x_test_imdb_small = x_test_imdb[:1000]
    y_test_imdb_small = y_test_imdb[:1000]
    
    # Создание модели
    model_imdb = create_model_imdb(max_features, maxlen, units1=64, units2=32)
    model_imdb.summary()
    
    # Обучение
    history_imdb, acc_imdb, loss_imdb = train_and_evaluate(
        model_imdb,
        x_train_imdb_small, y_train_imdb_small,
        x_test_imdb_small, y_test_imdb_small,
        epochs=5,
        batch_size=64,
        model_name="IMDB LSTM с кастомными слоями",
        is_binary=True
    )
    
    plot_training_history(history_imdb, "IMDB")
    
    # ========== ЧАСТЬ 3: ОБУЧЕНИЕ НА REUTERS ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 3: ОБУЧЕНИЕ НА REUTERS (МНОГОКЛАССОВАЯ КЛАССИФИКАЦИЯ)")
    print("="*60)
    
    # Загрузка данных
    max_features_reuters = 5000
    maxlen_reuters = 100
    (x_train_reuters, y_train_reuters), (x_test_reuters, y_test_reuters), num_classes = preprocess_reuters_data(max_features_reuters, maxlen_reuters)
    
    # Используем часть данных для ускорения
    x_train_reuters_small = x_train_reuters[:3000]
    y_train_reuters_small = y_train_reuters[:3000]
    x_test_reuters_small = x_test_reuters[:500]
    y_test_reuters_small = y_test_reuters[:500]
    
    # Создание модели
    model_reuters = create_model_reuters(max_features_reuters, maxlen_reuters, num_classes, units1=128, units2=64)
    model_reuters.summary()
    
    # Обучение
    history_reuters, acc_reuters, loss_reuters = train_and_evaluate(
        model_reuters,
        x_train_reuters_small, y_train_reuters_small,
        x_test_reuters_small, y_test_reuters_small,
        epochs=5,
        batch_size=64,
        model_name="Reuters LSTM с кастомными слоями",
        is_binary=False
    )
    
    plot_training_history(history_reuters, "Reuters")
    
    # ========== ЧАСТЬ 4: ГЛУБОКАЯ МОДЕЛЬ ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 4: ГЛУБОКАЯ МОДЕЛЬ С ТРЕМЯ LSTM СЛОЯМИ")
    print("="*60)
    
    model_deep = create_deep_lstm_model(max_features, maxlen)
    model_deep.summary()
    
    history_deep, acc_deep, loss_deep = train_and_evaluate(
        model_deep,
        x_train_imdb_small, y_train_imdb_small,
        x_test_imdb_small, y_test_imdb_small,
        epochs=5,
        batch_size=64,
        model_name="Глубокая LSTM модель",
        is_binary=True
    )
    
    plot_training_history(history_deep, "Deep_LSTM")
    
    # ========== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ==========
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЛАБОРАТОРНОЙ РАБОТЫ №4")
    print("="*60)
    
    print("\nРезультаты обучения:")
    print("-" * 50)
    print(f"  IMDB (бинарная классификация):")
    print(f"    - Точность: {acc_imdb:.4f} ({acc_imdb*100:.2f}%)")
    print(f"    - Ошибка: {loss_imdb:.4f}")
    print(f"\n  Reuters (многоклассовая классификация):")
    print(f"    - Точность: {acc_reuters:.4f} ({acc_reuters*100:.2f}%)")
    print(f"    - Ошибка: {loss_reuters:.4f}")
    print(f"\n  Глубокая LSTM модель (3 слоя):")
    print(f"    - Точность: {acc_deep:.4f} ({acc_deep*100:.2f}%)")
    print(f"    - Ошибка: {loss_deep:.4f}")
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("1. Кастомный LSTM слой успешно реализован и совместим с TensorFlow")
    print("2. Слой поддерживает все требуемые параметры:")
    print("   - Количество единиц памяти (units)")
    print("   - return_sequences (True/False)")
    print("   - Функции активации (activation и recurrent_activation)")
    print("3. Построена рекуррентная модель с двумя кастомными LSTM слоями")
    print("4. Модель успешно обучена на наборах данных IMDB и Reuters")
    print("5. Использованы оптимизатор Adam и функции потерь (binary/categorical crossentropy)")
    print("6. Реализована пошаговая обработка последовательности в методе call")
    print("7. Состояния h_t и c_t инициализируются нулями (или передаются явно)")
    print("="*60)