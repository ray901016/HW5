import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import os

# 確保僅使用 CPU
tf.config.set_visible_devices([], 'GPU')
print(f"Using device: {tf.test.gpu_device_name() if tf.config.list_physical_devices('GPU') else 'CPU'}")

# 加載 CIFAR-10 資料集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 歸一化數據到 [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 將標籤轉換為 One-Hot 編碼
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 資料增強
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# 建立一個簡單的 CNN 模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # CIFAR-10 有 10 個分類
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
print("Training model...")
model.fit(datagen.flow(X_train, y_train, batch_size=32),  # 減小 batch size
          validation_data=(X_test, y_test), epochs=10)

# 保存模型
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/cifar10_cnn.h5")
print("Model saved to 'saved_models/cifar10_cnn.h5'")

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
