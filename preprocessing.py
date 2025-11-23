import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np


AUTOTUNE = tf.data.AUTOTUNE

image_width = 86*4
image_height = 62*2
batch_size = 32 # best training was 32
padding_token = 99


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    # image = tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)

    return image


def preprocess_image(image_path, img_size=(image_width, image_height), augment=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, 0)
    image = distortion_free_resize(image, img_size)

    if augment:
        # Random horizontal flip
        # image = tf.image.random_flip_left_right(image)
        # Random shear transformation
        shear_x = np.random.uniform(-0.3, 0.3)
        shear_y = np.random.uniform(-0.3, 0.3)
        image = tfa.image.transform_ops.shear_x(image, level=shear_x, replace=0)
        image = tfa.image.transform_ops.shear_y(image, level=shear_y, replace=0)

 
        # Random brightness and contrast adjustment
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label, char_to_num, max_len):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length 
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)

    return label

def process_images_labels(image_path, label, augment=False):
    image = preprocess_image(image_path, augment=augment)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels, char_to_num, max_len, augment=False, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    dataset = dataset.map(
        lambda image_path, label: process_images_labels(
            image_path, label, char_to_num, max_len, augment=augment
        ),
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(batch_size).cache().prefetch(AUTOTUNE)
    return dataset

