import numpy as np
import tensorflow as tf
import cv2
import os
import uuid
from tensorflow.keras.preprocessing import image


def preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess image for the model."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap and return activation level."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Handle list or tensor outputs
        if isinstance(predictions, list):
            predictions = predictions[0]
        predictions = tf.convert_to_tensor(predictions)

        pred_index = tf.argmax(predictions[0])
        loss = predictions[0][pred_index]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val != 0:
            heatmap /= max_val
        else:
            heatmap = np.zeros_like(heatmap)


        # Compute average activation (how much of the image was highlighted)
    activation_level = float(np.mean(heatmap))

    return heatmap, activation_level


def save_and_overlay_heatmap(original_img_path, heatmap, output_dir='static/heatmaps', intensity=0.4):
    """Overlay heatmap on the original image and save it."""
    img = cv2.imread(original_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - intensity, heatmap_color, intensity, 0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"heatmap_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return output_path
