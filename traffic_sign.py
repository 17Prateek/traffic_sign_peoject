import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import os
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
path = r"C:\Users\HOME\Desktop\git\ImageColorDetector\traffic_Data\DATA"
labelfile = r"C:\Users\HOME\Desktop\git\ImageColorDetector\labels.csv"
batch_size_val = 32
epochs_val = 10
imageDimesions = (32, 32, 3)  # Target image dimensions
testRatio = 0.2
validationRatio = 0.2

count = 0
image = []
classNo = []
mylist = os.listdir(path)
print("Total classes detected:", len(mylist))

noofclasses = len(mylist)

for x in range(0, len(mylist)):
    mypiclist = os.listdir(os.path.join(path, str(count)))
    for y in mypiclist:
        curimg = cv2.imread(os.path.join(path, str(count), y))
        if curimg is not None:
            # Resize the image to the target size (32, 32)
            curimg = cv2.resize(curimg, (imageDimesions[0], imageDimesions[1]))
            image.append(curimg)
            classNo.append(count)
        else:
            print(f"Error reading image: {os.path.join(path, str(count), y)}")
    print(count, end=" ")
    count += 1

# Convert lists to numpy arrays
images = np.array(image)
classNo = np.array(classNo)

# Check the shape of the images array
print("Images array shape:", images.shape)
print("Class numbers array shape:", classNo.shape)

# Train-test split
X_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, x_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

data = pd.read_csv(labelfile)
print("Data shape:", data.shape, type(data))

# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize to range [0, 1]
    return img

# Apply preprocessing to images
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, x_validation)))
X_test = np.array(list(map(preprocessing, x_test)))

# Print shapes after preprocessing
print(f"Shape of X_train after preprocessing: {X_train.shape}")
print(f"Shape of X_validation after preprocessing: {X_validation.shape}")
print(f"Shape of X_test after preprocessing: {X_test.shape}")

# Reshape the data to add the channel dimension (1 for grayscale)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Data augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

# One-hot encoding for labels
y_train = to_categorical(y_train, noofclasses)
y_validation = to_categorical(y_validation, noofclasses)
y_test = to_categorical(y_test, noofclasses)

# Define the model
def mymodel():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noofclasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = mymodel()
print(model.summary())
history = model.fit(dataGen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) // 32,
                    epochs=epochs_val,
                    validation_data=(X_validation, y_validation),
                    shuffle=True)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test Accuracy:', score[1])

# Save the model
model.save("model.h5")




