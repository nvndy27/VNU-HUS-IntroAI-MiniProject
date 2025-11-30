# train.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import get_data_generators

IMG_SIZE = (224, 224)
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 30
MODEL_NAME = "mobilenetv2"

def build_model(model_name="mobilenetv2"):
    if model_name.lower() == "mobilenetv2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3))
    else:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3))

    base_model.trainable = False  # freeze base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_data_generators()
    model = build_model(MODEL_NAME)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint("model/model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, earlystop]
    )

    print("Training finished!")

    import matplotlib.pyplot as plt

    # Vẽ Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    plt.show()

    # Vẽ Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()
