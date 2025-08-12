# app.py
import io
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------
# App Config
# --------------------------------------------
st.set_page_config(page_title="AI Medical Diagnosis", layout="centered")
st.title("🧬 AI Medical Diagnosis System")
st.write("Upload an X-ray or skin image to detect possible diseases using AI.")

# --------------------------------------------
# Load Model
# --------------------------------------------
@st.cache_resource
def load_model():
    model_path = "model/model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please place your model at: `model/model.h5`")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# ⚠️ Update this list to match your model’s output layer
class_names = [
    "Pneumonia",
    "Normal",
    "Covid-19",
    "Lung Opacity",
    "Tuberculosis",
    "Fibrosis",
    "Emphysema"
]


# --------------------------------------------
# Preprocess Function
# --------------------------------------------
def preprocess_image(uploaded_file, input_shape):
    """
    Preprocess the uploaded image to match the input shape expected by the model.
    Supports both grayscale and RGB models.
    """
    try:
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes))

        # Ensure the image has 3 channels if needed
        if input_shape[-1] == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        image = image.resize((input_shape[0], input_shape[1]))
        image_array = np.array(image, dtype=np.float32) / 255.0

        if input_shape[-1] == 1:
            image_array = np.expand_dims(image_array, axis=-1)  # (H, W, 1)

        image_array = np.expand_dims(image_array, axis=0)  # (1, H, W, C)
        return image_array
    except Exception as e:
        st.error(f"❌ Error during image preprocessing: {e}")
        return None

# --------------------------------------------
# File Upload & Prediction
# --------------------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if model:
        with st.spinner("🔍 Analyzing..."):
            try:
                input_shape = model.input_shape[1:]  # e.g. (224, 224, 3)
                image_array = preprocess_image(uploaded_file, input_shape)

                if image_array is None:
                    st.stop()

                prediction = model.predict(image_array)
                prediction = prediction[0]
                st.write(prediction.shape)
                print(prediction)
                # Validate prediction output length

                if len(prediction) != len(class_names):
                    raise ValueError("Mismatch between model output and class_names length.")(
                    f"Mismatch: Model returned {len(prediction)} outputs but class_names has {len(class_names)} items."
                    )

                confidence = np.max(prediction)
                predicted_class = class_names[np.argmax(prediction)]

                top_index = int(np.argmax(prediction))
                confidence = float(prediction[top_index]) * 100
                diagnosis = class_names[top_index]

                # Output results
                st.subheader(f"🩺 Prediction: **{diagnosis}**")
                st.write(f"🔢 Confidence: **{confidence:.2f}%**")

                if confidence > 80:
                    st.success("✅ High confidence diagnosis. Please consult a physician for confirmation.")
                else:
                    st.warning("⚠️ Low confidence - consider retaking or using a clearer image.")

            except Exception as e:
                st.error(f"❌ Image preprocessing or prediction failed: {e}")
    else:
        st.error("🚫 Model not loaded.")
