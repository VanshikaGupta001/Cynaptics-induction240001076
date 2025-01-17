import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

train = '/kaggle/input/newinductiondataset/Data/Train'
test = '/kaggle/input/induction-task-2025/Test_Images'

# Preprocessing the data
datagen = ImageDataGenerator(rescale=1./255)  # Removed validation_split

train_generator = datagen.flow_from_directory(
    train,
    target_size=(256, 256),
    batch_size=15,
    class_mode='binary'
)
    
# Building the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=train_generator.samples // train_generator.batch_size  
)

predictions = []
test_images = os.listdir(test)

# Making predictions on test data
for image_name in test_images:
    try:
        image_path = os.path.join(test, image_name)
        image = load_img(image_path, target_size=(256, 256))  # Resize
        image_array = img_to_array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        prediction = model.predict(image_array)
        label = 'AI' if prediction[0] < 0.5 else 'Real'
        predictions.append((image_name, label))
    except Exception as e:
        print(f"Error {image_name}:{e}")
        continue

# Submission
submission_df = pd.DataFrame(predictions, columns=['Id', 'Label'])
submission_path = 'submission.csv' 
submission_df.to_csv(submission_path, index=False)

print("saved")
