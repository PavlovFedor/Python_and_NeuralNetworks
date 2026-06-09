import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from model import build_model

# Убираем предупреждения
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# 1. Создаём датасет
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_classes=2,
    random_state=42
)

# 2. Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Нормализуем данные
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Строим модель
model = build_model()

# 5. Компилируем
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # pyright: ignore[reportAttributeAccessIssue]
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. Настраиваем EarlyStopping для предотвращения переобучения
early_stopping = tf.keras.callbacks.EarlyStopping( # type: ignore
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1,
    mode='min'  # минимизируем val_loss
)

# 7. Обучаем
print("Начало обучения...")
history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Оцениваем на тестовых данных
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nРезультаты на тестовой выборке:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")

# 9. Выводим архитектуру модели
print("\nАрхитектура модели:")
model.summary()

# 10. Выводим статистику обучения
print(f"\nВсего эпох: {len(history.history['loss'])}")
print(f"Лучшая val_loss: {min(history.history['val_loss']):.4f}")
print(f"Лучшая val_accuracy: {max(history.history['val_accuracy']):.4f}")