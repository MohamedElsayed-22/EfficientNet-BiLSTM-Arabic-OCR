import keras
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, Model

class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

def build_model(image_width, image_height, num_classes):
    # Preprocessing for EfficientNetB3
    input_img = keras.Input(shape=(image_width, image_height, 3), name="image", dtype="float32")
    labels = keras.layers.Input(name="label", shape=(None,), dtype="float32")

    # EfficientNetB3 as the base model
    base_model = EfficientNetB3(include_top=False, input_tensor=input_img, weights="imagenet")
    for layer in base_model.layers:
        layer.trainable = True

    # Reshape for RNN layers
    output_shape = base_model.output_shape
    h, w, c = output_shape[1], output_shape[2], output_shape[3]
    new_shape = (h * w, c)
    x = layers.Reshape(target_shape=new_shape)(base_model.output)

    # RNN Layers
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Self-Attention Layer
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=128)(x, x)
    x = layers.Concatenate()([x, attention])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.LayerNormalization()(x)

    # Dense Layer
    x = layers.Dense(num_classes, activation="softmax", name="dense3")(x)

    # CTC Layer
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Model Definition
    model = Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")

    # Optimizer and Learning Rate Scheduler
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001, first_decay_steps=5000, t_mul=2.0, m_mul=0.9, alpha=0.01
    )
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt)
    print(model.summary())

    return model



