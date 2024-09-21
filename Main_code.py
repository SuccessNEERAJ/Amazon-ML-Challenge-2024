import os
import pandas as pd

# Define dataset folder path
DATASET_FOLDER = '/content/ML/dataset'

# Load datasets (train, test, and sample test sets)
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
sample_test_out = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test_out.csv'))

# Append utils script to system path to import custom functions
import sys
sys.path.append(os.path.dirname(os.path.abspath("/content/ML/src/utils.py")))
from utils import download_images, parse_string

# Import necessary libraries for model building, image processing, and prediction
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm  # For progress bars
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import gc  # For garbage collection to free memory
from concurrent.futures import ThreadPoolExecutor  # For concurrent image processing
from pathlib import Path
import time

# Append additional scripts to system path and import custom constants and sanity check
sys.path.append(os.path.dirname(os.path.abspath("/content/ML/src/constants.py")))
sys.path.append(os.path.dirname(os.path.abspath("/content/ML/src/sanity.py")))

from constants import entity_unit_map, allowed_units
from sanity import sanity_check

# Constants for image processing
IMG_SIZE = (224, 224)  # Resize images to 224x224
BATCH_SIZE = 32  # Batch size for model training

# Function to load and preprocess images from URL
def load_image_from_url(url):
    try:
        # Send a GET request to download the image
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')  # Convert to RGB
        img = img.resize(IMG_SIZE)  # Resize the image
        img = np.array(img)
        img = preprocess_input(img)  # Preprocess image for ResNet50 model
        return img
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

# Preprocess the entity value labels into categorical groups
def preprocess_label(label):
    label = label.lower()  # Convert label to lowercase
    # Group labels into general categories
    if 'gram' in label or 'weight' in label:
        return 'weight'
    elif 'height' in label:
        return 'height'
    elif 'width' in label:
        return 'width'
    elif 'depth' in label:
        return 'depth'
    elif 'volume' in label:
        return 'volume'
    elif 'voltage' in label:
        return 'voltage'
    elif 'wattage' in label:
        return 'wattage'
    else:
        return 'other'  # Fallback category for unknown labels

# Build a simple Convolutional Neural Network model for image classification
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # First Conv layer
        MaxPooling2D((2, 2)),  # First MaxPool layer
        Conv2D(64, (3, 3), activation='relu'),  # Second Conv layer
        MaxPooling2D((2, 2)),  # Second MaxPool layer
        Conv2D(64, (3, 3), activation='relu'),  # Third Conv layer
        Flatten(),  # Flatten layer for fully connected layers
        Dense(64, activation='relu'),  # Dense hidden layer
        Dropout(0.5),  # Dropout to prevent overfitting
        Dense(num_classes, activation='softmax')  # Output layer for classification
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Download, process, and predict the entity value from an image using the model
def download_and_process_image(row, model, label_encoder):
    image_url = row['image_link']  # Get image URL from the row
    img = load_image_from_url(image_url)  # Download and preprocess the image

    if img is not None:
        img = np.expand_dims(img, axis=0)  # Add batch dimension for model input
        y_pred = model.predict(img)  # Predict the label for the image
        y_pred_class = np.argmax(y_pred, axis=1)  # Get the class with the highest score
        predicted_label = label_encoder.inverse_transform(y_pred_class)  # Decode predicted label

        # Get the entity name from the row and map it to the appropriate unit
        entity_name = row.get('entity_name', '')
        if entity_name in entity_unit_map and entity_unit_map[entity_name]:
            value = np.random.uniform(0, 10)  # Random prediction for demonstration
            unit = next(iter(entity_unit_map[entity_name]))  # Select the first allowed unit
            formatted_prediction = f"{value:.2f} {unit}"  # Format prediction as value + unit
        else:
            formatted_prediction = ""  # Empty prediction if entity name is not valid

        # Clean up memory after processing
        del img
        gc.collect()

        return formatted_prediction
    else:
        return ""  # Return empty string if image download fails

# Process images in parallel using multiple threads to speed up predictions
def preprocess_images_concurrently(test_df, model, label_encoder):
    predictions = []

    # Use ThreadPoolExecutor to download and process images concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(lambda row: download_and_process_image(row, model, label_encoder),
                                         [row for _, row in test_df.iterrows()]), total=len(test_df)))

    predictions.extend(results)
    return predictions

# Train the CNN model with the train dataset
def train_model():
    # Load a subset of training data (for demonstration, you can load the entire dataset)
    train_df = pd.read_csv('/content/ML/dataset/train.csv').head(1000)

    images, labels = [], []  # Initialize lists to store images and labels
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_url = row['image_link']
        label = preprocess_label(row['entity_value'])  # Preprocess the label
        img = load_image_from_url(image_url)  # Download and preprocess the image
        if img is not None:
            images.append(img)  # Append image to list
            labels.append(label)  # Append label to list

            # Clear memory after processing each image to avoid memory overflow
            del img
            gc.collect()

    # Encode labels into numerical format
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, y_encoded, test_size=0.2, random_state=42)

    # Resample the training set to handle class imbalance
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(np.array(X_train).reshape(len(X_train), -1), y_train)
    X_train_resampled = X_train_resampled.reshape(-1, 224, 224, 3)

    # Convert labels to categorical format for the model
    y_train_categorical = to_categorical(y_train_resampled)
    y_val_categorical = to_categorical(y_val)

    # Create and compile the CNN model
    model = create_model((224, 224, 3), len(label_encoder.classes_))
    
    # Define callbacks for learning rate reduction and early stopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train_resampled, y_train_categorical,
                        validation_data=(np.array(X_val), y_val_categorical),
                        epochs=100,
                        batch_size=32,
                        callbacks=[reduce_lr, early_stopping])

    return model, label_encoder

# Main function to execute the workflow
def main():
    model, label_encoder = train_model()  # Train the model and get label encoder

    # Load test data (for demonstration, only 1000 samples)
    test_df = pd.read_csv('/content/ML/dataset/test.csv').head(1000)

    # Process images and make predictions concurrently
    predictions = preprocess_images_concurrently(test_df, model, label_encoder)

    # Save predictions to a CSV file
    output_df = pd.DataFrame({
        'index': test_df['index'],
        'prediction': predictions
    })
    output_df.to_csv('predictions.csv', index=False)

    # Run a sanity check to validate the predictions
    try:
        sanity_check('/content/ML/dataset/test.csv', 'predictions.csv')
        print("Sanity check passed successfully!")
    except Exception as e:
        print(f"Sanity check failed: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()