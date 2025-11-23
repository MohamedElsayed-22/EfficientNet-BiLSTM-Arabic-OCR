from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import *
from preprocessing import *
from model import *
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import json

np.random.seed(42)
tf.random.set_seed(42)

AUTOTUNE = tf.data.AUTOTUNE


base_dir = "Datasets/khatt"

train_images, train_labels = get_dataset_paths_and_filenames(base_dir, "train")
test_images, test_labels = get_dataset_paths_and_filenames(base_dir, "test")
validate_images, validate_labels = get_dataset_paths_and_filenames(base_dir, "validate")

train_labels_cleaned = []
characters = set()
train_labels_cleaned, max_len_1 = clean_labels(train_labels, characters, train=True)
validation_labels_cleaned, max_len_2 = clean_labels(validate_labels, characters)
test_labels_cleaned, max_len_3 = clean_labels(test_labels, characters)

max_len = max(max_len_2, max_len_1)
max_len = max(max_len_3, max_len)

# Mappings
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

num_to_char_dict = dict(enumerate(num_to_char.get_vocabulary()))

# Save to a JSON file
with open('num_to_char.json', 'w') as file:
    json.dump(num_to_char_dict, file)

label = vectorize_label(train_labels_cleaned[0], num_to_char, max_len)
indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
label = tf.strings.reduce_join(num_to_char(indices))
label = label.numpy().decode("utf-8")

train_ds = prepare_dataset(train_images, train_labels_cleaned, char_to_num, max_len, augment=True, shuffle=True)
validation_ds = prepare_dataset(validate_images, validation_labels_cleaned, char_to_num, max_len)
test_ds = prepare_dataset(test_images, test_labels_cleaned, char_to_num, max_len)


validation_images = [batch["image"] for batch in validation_ds]
validation_labels = [batch["label"] for batch in validation_ds]



num_classes = len(char_to_num.get_vocabulary()) + 2  
model = build_model(image_width, image_height, num_classes)

epochs = 100  

best_model_path = "best_model_khatt.h5"

# Check if checkpoint exists
if os.path.exists(best_model_path):
    print("Loading weights from saved checkpoint.")
    model.load_weights(best_model_path)
else:
    print("No checkpoint found. Starting training from scratch.")

# Callbacks setup
model_checkpoint_callback = ModelCheckpoint(
    filepath=best_model_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
    )

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense3").output
    )
                                 
print("input: ", model.get_layer(name="image").input)
print("output: ", model.get_layer(name="dense3").output)

edit_distance_callback = EditDistanceCallback(prediction_model, validation_images, validation_labels)

stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    )

# Train the model.
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[model_checkpoint_callback, edit_distance_callback],
    shuffle=True,
    )

last_model_path = "last_model_khatt.h5"
model.save(last_model_path)
