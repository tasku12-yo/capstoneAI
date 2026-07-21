import os
import sys

print("🚀 Memulai proses training...")

# 1. Konfigurasi Path & Hyperparameter
DATASET_DIR = 'egg/'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

# Cek keberadaan folder
if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    print(f"❌ Error: Folder '{TRAIN_DIR}' atau '{TEST_DIR}' tidak ditemukan!")
    print("Pastikan struktur folder kamu: dataset/train/ dan dataset/test/")
    sys.exit(1)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 3

print(" Memuat ImageDataGenerator...")

# Data Augmentation untuk data training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Rescale saja untuk data test (validasi)
test_datagen = ImageDataGenerator(rescale=1./255)

try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Memakai folder 'test' sebagai data validasi
    val_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
except Exception as e:
    print(f"❌ Error saat memuat gambar: {e}")
    sys.exit(1)

print("\nUrutan Indeks Kelas:", train_generator.class_indices)

# 2. Membangun Arsitektur EfficientNetB0
print("\n Membangun Arsitektur Model...")
base_model = EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Setup Callback & Path Penyimpanan Model
os.makedirs('models', exist_ok=True)
checkpoint_filepath = 'models/egg_model.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# 4. Eksekusi Training
print("\n Mulai Training Model...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint_callback]
)

print(f"\n✅ Training Selesai! Model terbaik tersimpan di: {checkpoint_filepath}")