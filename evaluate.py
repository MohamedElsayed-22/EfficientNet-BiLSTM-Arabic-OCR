from train import *

with open('num_to_char.json', 'r') as file:
    num_to_char_dict = json.load(file)

num_to_char_vocabulary = [num_to_char_dict[str(i)] for i in range(len(num_to_char_dict))]
num_to_char = StringLookup(vocabulary=num_to_char_vocabulary, mask_token=None, invert=True)


trained_model = keras.models.load_model('best_model_khatt.h5', custom_objects={'CTCLayer': CTCLayer})

# Create a prediction model from the loaded model
prediction_model_loaded = keras.models.Model(
    trained_model.get_layer(name="image").input, 
    trained_model.get_layer(name="dense3").output
)

total_correct_predictions = 0
total_true_letters = 0
num_of_test_samples = 0

for batch in test_ds:
    batch_images = batch["image"]
    labels = batch["label"]
    print("batch len:", len(batch["label"]))
    num_of_test_samples += len(batch["label"])

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)


    for i in range(len(batch["label"])):
        img = batch_images[i]
        
        label = labels[i]
        indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
        # Convert to string.
        label = tf.strings.reduce_join(num_to_char(indices))
        label_str = label.numpy().decode("utf-8")

        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]
        reflected_img = np.fliplr(img)

        title = f"Prediction: {pred_texts[i][::-1]}"

        # Convert label to string for comparison
        true_label_str = tf.strings.reduce_join(tf.strings.as_string(label)).numpy().decode('utf-8') if label is not None else ""

        # Calculate accuracy for each prediction separately
        correct_predictions = sum([1 for pred_char, true_char in zip(pred_texts[i], true_label_str) if pred_char == true_char])
        total_true_letters += len(true_label_str)
        total_correct_predictions += correct_predictions

        print("pred: ", pred_texts[i]) 
        print("true label: ", true_label_str)
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Total True Letters: {len(true_label_str)}")
        print(f"Prediction Accuracy: {correct_predictions / max(len(true_label_str), 1) * 100:.2f}%")


overall_accuracy = total_correct_predictions / total_true_letters
print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}% of samples: {num_of_test_samples}")

