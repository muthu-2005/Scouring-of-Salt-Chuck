import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import keras
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load and preprocess images
labels = ['no-plastic', 'plastic']
X_train = []
y_train = []
image_size = 224

for i in labels:
    folderPath = os.path.join(r'/content/drive/MyDrive/Liveinlab', 'train', i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)

for i in labels:
    folderPath = os.path.join(r'/content/drive/MyDrive/Liveinlab', 'test', i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train, y_train = shuffle(X_train, y_train, random_state=101)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=101)

# Convert labels to categorical
y_train = [labels.index(i) for i in y_train]
y_train = tf.keras.utils.to_categorical(y_train)

y_test = [labels.index(i) for i in y_test]
y_test = tf.keras.utils.to_categorical(y_test)

# Define a more complex CNN model
def create_model():
    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs, x)
    return model

model = create_model()

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint('/content/ssc_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Train model
r = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32, verbose=1, callbacks=[es, checkpoint, reduce_lr])

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
model.save('ssc_model.h5' )
# Load model for prediction
model = tf.keras.models.load_model('/content/ssc_model.h5')

# Preprocess image for prediction
def preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Example usage
image_path = '/content/Example.jpeg'
target_size = (224, 224)
image_array = preprocess_image(image_path, target_size)

# Make predictions
predictions = model.predict(image_array)
predicted_class_index = np.argmax(predictions, axis=1)

# Map the predicted class index to class names
class_names = ['no-plastic', 'plastic']
predicted_class = class_names[predicted_class_index[0]]

print(f'Predicted class: {predicted_class}')
