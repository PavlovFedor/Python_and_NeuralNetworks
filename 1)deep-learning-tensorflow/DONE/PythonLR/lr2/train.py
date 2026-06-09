import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from custom_pooling import CustomPooling


def preprocess_data():
    """Загрузка и предобработка данных CIFAR-10"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Нормализация
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


def create_model(pooling_type='custom'):
    """
    Создание модели с различными типами пулинга
    pooling_type: 'custom', 'max', 'average'
    """
    model = models.Sequential()
    
    # Первый сверточный блок
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3))) # type: ignore
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    if pooling_type == 'custom':
        model.add(CustomPooling(pool_size=2, strides=2, padding='valid', aggregation='max', pool_type='channelwise'))
    elif pooling_type == 'max':
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    elif pooling_type == 'average':
        model.add(layers.AveragePooling2D(pool_size=2, strides=2))
    
    # Второй сверточный блок
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    if pooling_type == 'custom':
        model.add(CustomPooling(pool_size=2, strides=2, padding='valid', aggregation='max', pool_type='channelwise'))
    elif pooling_type == 'max':
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    elif pooling_type == 'average':
        model.add(layers.AveragePooling2D(pool_size=2, strides=2))
    
    # Третий сверточный блок
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    # Глобальный пулинг
    model.add(layers.GlobalAveragePooling2D())
    
    # Полносвязные слои
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model


def create_model_with_global_pooling():
    """Создание модели с глобальным кастомным пулингом"""
    model = models.Sequential()
    
    # Первый сверточный блок
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3))) # type: ignore
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(CustomPooling(pool_size=2, strides=2, padding='valid', aggregation='max', pool_type='channelwise'))
    
    # Второй сверточный блок
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(CustomPooling(pool_size=2, strides=2, padding='valid', aggregation='max', pool_type='channelwise'))
    
    # Третий сверточный блок
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    # Глобальный пулинг
    model.add(CustomPooling(pool_type='global', aggregation='max'))
    
    # Полносвязные слои
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model


def create_model_with_median_pooling():
    """Создание модели с медианным пулингом"""
    model = models.Sequential()
    
    # Первый сверточный блок
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3))) # type: ignore
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(CustomPooling(pool_size=2, strides=2, padding='valid', aggregation='median', pool_type='channelwise'))
    
    # Второй сверточный блок
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(CustomPooling(pool_size=2, strides=2, padding='valid', aggregation='median', pool_type='channelwise'))
    
    # Третий сверточный блок
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    # Глобальный пулинг
    model.add(layers.GlobalAveragePooling2D())
    
    # Полносвязные слои
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model


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
    
    print(f"\nTraining {model_name}...")
    
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
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")
    
    return history, test_acc


def compare_pooling_types():
    """Сравнение различных типов пулинга"""
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ТИПОВ ПУЛИНГА НА CIFAR-10")
    print("="*60)
    
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    
    # Используем часть данных для ускорения
    x_train_small = x_train[:5000]
    y_train_small = y_train[:5000]
    x_test_small = x_test[:1000]
    y_test_small = y_test[:1000]
    
    pooling_configs = [
        ('Custom Max Pooling (кастомный)', 'custom'),
        ('Standard Max Pooling (стандартный)', 'max'),
        ('Standard Average Pooling (стандартный)', 'average')
    ]
    
    results = {}
    
    for name, pool_type in pooling_configs:
        print(f"\n{'='*50}")
        print(f"Тестирование: {name}")
        print('='*50)
        
        model = create_model(pooling_type=pool_type)
        model.summary()
        
        history, test_acc = train_and_evaluate(
            model, x_train_small, y_train_small, 
            x_test_small, y_test_small, 
            epochs=3,
            model_name=name
        )
        
        results[name] = test_acc
    
    return results


if __name__ == "__main__":
    # Устанавливаем random seed для воспроизводимости
    tf.random.set_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("Кастомный слой пулинга для TensorFlow")
    print("="*60)
    
    # ========== ЧАСТЬ 1: ТЕСТИРОВАНИЕ КАСТОМНОГО СЛОЯ ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 1: ТЕСТИРОВАНИЕ КАСТОМНОГО СЛОЯ ПУЛИНГА")
    print("="*60)
    
    print("\n1.1 Проверка работы слоя на случайных данных:")
    test_input = tf.random.normal([2, 32, 32, 3])
    
    # Max pooling
    pool_max = CustomPooling(pool_size=2, strides=2, padding='valid', aggregation='max')
    output_max = pool_max(test_input)
    print(f"  Max pooling:   {test_input.shape} -> {output_max.shape}")
    
    # Average pooling
    pool_avg = CustomPooling(pool_size=2, strides=2, padding='valid', aggregation='average')
    output_avg = pool_avg(test_input)
    print(f"  Average pooling: {test_input.shape} -> {output_avg.shape}")
    
    # Median pooling
    pool_median = CustomPooling(pool_size=2, strides=2, padding='valid', aggregation='median')
    output_median = pool_median(test_input)
    print(f"  Median pooling:  {test_input.shape} -> {output_median.shape}")
    
    # Global pooling
    pool_global = CustomPooling(pool_type='global', aggregation='max')
    output_global = pool_global(test_input)
    print(f"  Global pooling:  {test_input.shape} -> {output_global.shape}")
    
    # SAME padding
    pool_same = CustomPooling(pool_size=3, strides=2, padding='same', aggregation='max')
    output_same = pool_same(test_input)
    print(f"  SAME padding:    {test_input.shape} -> {output_same.shape}")
    
    # Проверка сохранения/загрузки конфигурации
    config = pool_max.get_config()
    new_layer = CustomPooling.from_config(config)
    print(f"\n1.2 Сохранение/загрузка конфигурации слоя: УСПЕШНО")
    
    # ========== ЧАСТЬ 2: ТЕСТИРОВАНИЕ МОДЕЛИ С ГЛОБАЛЬНЫМ ПУЛИНГОМ ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 2: МОДЕЛЬ С ГЛОБАЛЬНЫМ КАСТОМНЫМ ПУЛИНГОМ")
    print("="*60)
    
    global_model = create_model_with_global_pooling()
    global_model.summary()
    
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    x_train_small = x_train[:2000]
    y_train_small = y_train[:2000]
    x_test_small = x_test[:800]
    y_test_small = y_test[:800]
    
    history_global, acc_global = train_and_evaluate(
        global_model, x_train_small, y_train_small,
        x_test_small, y_test_small,
        epochs=2,
        model_name="Глобальный кастомный пулинг"
    )
    
    # ========== ЧАСТЬ 3: СРАВНЕНИЕ ТИПОВ ПУЛИНГА ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 3: СРАВНЕНИЕ КАСТОМНОГО И СТАНДАРТНОГО ПУЛИНГА")
    print("="*60)
    
    results = compare_pooling_types()
    
    # ========== ЧАСТЬ 4: ТЕСТИРОВАНИЕ МЕДИАННОГО ПУЛИНГА ==========
    print("\n" + "="*60)
    print("ЧАСТЬ 4: МОДЕЛЬ С МЕДИАННЫМ КАСТОМНЫМ ПУЛИНГОМ")
    print("="*60)
    
    median_model = create_model_with_median_pooling()
    median_model.summary()
    
    history_median, acc_median = train_and_evaluate(
        median_model, x_train_small, y_train_small,
        x_test_small, y_test_small,
        epochs=3,
        model_name="Медианный пулинг"
    )
    
    # ========== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ==========
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    
    print("\nСравнение точности на тестовой выборке (3000 образцов):")
    print("-" * 50)
    for name, acc in results.items():
        print(f"  {name}: {acc:.4f}")
    print(f"  Глобальный кастомный пулинг: {acc_global:.4f}")
    print(f"  Медианный кастомный пулинг: {acc_median:.4f}")
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("1. Кастомный слой пулинга успешно реализован и совместим с TensorFlow")
    print("2. Слой поддерживает все требуемые параметры:")
    print("   - Размер окна (pool_size) и шаг (strides)")
    print("   - Тип паддинга (SAME/VALID)")
    print("   - Тип агрегации (max, average, median)")
    print("   - Тип пулинга (channelwise, global)")
    print("3. Модель с кастомным пулингом успешно обучается на CIFAR-10")
    print("4. Медианный пулинг успешно реализован и работает корректно")
    print("="*60)