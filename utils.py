import os
from tensorflow import keras
import tensorflow as tf
import numpy as np

def get_dataset_paths_and_filenames(base_dir, dataset_type):
    dataset_path = os.path.join(base_dir, dataset_type)
    image_path = os.path.join(dataset_path, "images")
    gt_path = os.path.join(dataset_path, "labels")

    filenames_img = sorted(os.listdir(image_path))
    filenames_gt = sorted(os.listdir(gt_path))

    filenames_img_split = [filename.replace('.jpeg', '') for filename in filenames_img]
    filenames_gt_split = [filename.replace('.txt', '') for filename in filenames_gt]

    images, labels = get_image_paths_and_labels(filenames_img_split, filenames_gt_split, image_path, gt_path)
    return images, labels


def get_image_paths_and_labels(filenames_img, filenames_labels, base_image_path, base_GT_path):
    images = []
    labels = []
    for sample in range(len(filenames_img)):
        img_path = os.path.join(base_image_path, filenames_img[sample] + '.jpeg')
        if os.path.getsize(img_path):
            label_path = os.path.join(base_GT_path, filenames_labels[sample] + '.txt')
            with open(label_path, "r", encoding='utf-8') as label_file:
                label = label_file.read()
            images.append(img_path)
            labels.append(label)
    
    return images, labels

def clean_labels(labels, characters, train=False):
    max_len = 0
    cleaned_labels = []
    
    for label in labels:
        label = label.strip()
        max_len = max(max_len, len(label))
        cleaned_labels.append(label)       
        if train:
            for char in label:
                characters.add(char)

    return cleaned_labels, max_len


def calculate_edit_distance(labels, predictions, max_len):    
    saprse_labels = tf.sparse.from_dense(labels)
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(  
        predictions, input_length=input_len, greedy=False, beam_width=100,
    )[0][0][:, :max_len]
    sparse_predictions =tf.sparse.from_dense(predictions_decoded)    
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model, validation_images, validation_labels):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_images = validation_images
        self.validation_labels = validation_labels

    def on_epoch_end(self, epoch, validation_images, validation_labels, logs=None):
        edit_distances = []
        for sample in range(len(validation_images)):
            labels = validation_labels[sample]
            predictions = self.prediction_model.predict(validation_images[sample])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}")


def decode_batch_predictions(pred, max_len, num_to_char):
    """
    A utility function to decode the output of the network.

    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
        ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    print(output_text)
    return output_text