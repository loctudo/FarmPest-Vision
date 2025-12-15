# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

import matplotlib.pyplot as plt

# ==============================
# 1. KIỂM TRA TENSORFLOW & GPU
# ==============================
print("TensorFlow version:", tf.__version__)

physical_gpus = tf.config.list_physical_devices('GPU')
logical_gpus  = tf.config.list_logical_devices('GPU')
print("Physical GPUs:", physical_gpus)
print("Logical GPUs :", logical_gpus)

base_directory = "C:/Users/Admin/Desktop/FarmPest Vision/dataset"

lables_names = sorted([d for d in os.listdir(base_directory)
                         if os.path.isdir(os.path.join(base_directory, d))])

batch_size = 128
# Training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_directory,
    validation_split=0.15,
    subset="training",
    seed=64,
    image_size=(300, 225),
    batch_size=batch_size,
    label_mode='categorical'  # One-hot encoded labels
)

# Validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_directory,
    validation_split=0.15,
    subset="validation",
    seed=64,
    image_size=(300, 225),
    batch_size=batch_size,
    label_mode='categorical'
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
], name="data_augmentation")

model = models.Sequential()

# Define the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(300, 225, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of the base model
model = models.Sequential([
    base_model,
    data_augmentation,
    layers.Rescaling(1./255),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(12, activation='softmax')
])

model.summary()

# Compile the model with optimizer, loss, metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
EPOCHS = 100

# Callbacks for better training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Save the model
model.save('pest_classification_model.keras')
print("Model saved as 'pest_classification_model.keras'")
model.save("pest_model.h5")
print("Model saved as 'pest_model.h5'")


# Lấy tập tên lớp (tên loài côn trùng)
class_names = train_ds.class_names
print("Class names:", class_names)

# Lấy 1 batch ảnh từ validation dataset
images_batch, labels_batch = next(iter(val_ds))   # (batch_size, 300,225,3)

# Chọn ngẫu nhiên 1 ảnh trong batch
idx = random.randint(0, images_batch.shape[0] - 1)

image = images_batch[idx].numpy().astype("uint8")    # ảnh gốc
label_onehot = labels_batch[idx].numpy()             # one-hot
true_index = np.argmax(label_onehot)                 # index lớp thật
true_class = class_names[true_index]                 # tên lớp thật

# ----- HIỂN THỊ ẢNH -----
plt.imshow(image)
plt.title(f"Actual: {true_class}  (class {true_index})")
plt.axis("off")
plt.show()

# ----- CHUẨN HÓA ẢNH ĐỂ DỰ ĐOÁN -----
# Nếu model có layers.Rescaling(1/255) thì dùng ảnh gốc luôn:
input_img = tf.expand_dims(image, axis=0)

# Nếu model KHÔNG có Rescaling thì dùng dòng dưới:
# input_img = tf.expand_dims(image/255.0, axis=0)

# ----- DỰ ĐOÁN -----
prediction = model.predict(input_img)
pred_index = np.argmax(prediction)
pred_class = class_names[pred_index]

print("===============================")
print("Actual insect:       ", true_class)
print("Predicted insect:    ", pred_class)
print("Actual class index:  ", true_index)
print("Predicted class idx: ", pred_index)
print("===============================")