# preprocess.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = "C:/Users/ASUS/Documents/AI_Project/PJ2/dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_data_generators():
    """Tạo generator cho train, val, test"""
    # Data augmentation chỉ áp dụng cho train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen

if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_data_generators()
    print("Data generators ready!")

