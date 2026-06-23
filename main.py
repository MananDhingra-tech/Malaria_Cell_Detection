import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# CONFIGURATION

DATA_DIR = r"cell_images"  
IMG_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 32
MODEL_NAME = "malaria_cell_detection_model.h5"

# DATA LOADING & PREPROCESSING

print("Loading dataset ")

PARASITIZED = os.path.join(DATA_DIR, "Parasitized")
UNINFECTED  = os.path.join(DATA_DIR, "Uninfected")

data, labels = [], []

# category 0 = Parasitized, 1 = Uninfected
for category, folder in enumerate([PARASITIZED, UNINFECTED]):
    print(f"Loading images from: {folder}")
    for img_name in tqdm(os.listdir(folder)):
        path = os.path.join(folder, img_name)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(category)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels, dtype="int32")

print(f"Dataset loaded: {data.shape[0]} images.")
print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
print(f" Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# DATA AUGMENTATION

datagen = ImageDataGenerator(
    rotation_range=10,        # random rotation
    width_shift_range=0.1,    # horizontal shift
    height_shift_range=0.1,   # vertical shift
    horizontal_flip=True,     # random flips
    zoom_range=0.1            # random zoom
)
datagen.fit(X_train)

# MODEL ARCHITECTURE

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())


# TRAINING

print("Training model")
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2)
]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks = callbacks
)

# EVALUATION
print("Evaluating Model")

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Parasitized", "Uninfected"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Parasitized","Uninfected"],
            yticklabels=["Parasitized","Uninfected"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# TRAINING CURVES
print("Accuracy and loss curves")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Model Accuracy'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss'); plt.legend()
plt.tight_layout()
plt.show()

# SAVE MODEL
print(f"Saving model as '{MODEL_NAME}'")
model.save(MODEL_NAME)
print(f"Model saved {MODEL_NAME}")


