# Plant Disease Detection - Streamlit Demo
# Run:  streamlit run app.py   (keep best_model.keras in the same folder)

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image

IMG = 160
MODEL_PATH = "best_model.keras"

CLASS_NAMES = [
    "Apple - Apple scab", "Apple - Black rot", "Apple - Cedar apple rust", "Apple - healthy",
    "Blueberry - healthy", "Cherry - Powdery mildew", "Cherry - healthy",
    "Corn - Cercospora leaf spot (Gray leaf spot)", "Corn - Common rust", "Corn - Northern Leaf Blight",
    "Corn - healthy", "Grape - Black rot", "Grape - Esca (Black Measles)",
    "Grape - Leaf blight (Isariopsis Leaf Spot)", "Grape - healthy",
    "Orange - Huanglongbing (Citrus greening)", "Peach - Bacterial spot", "Peach - healthy",
    "Pepper bell - Bacterial spot", "Pepper bell - healthy", "Potato - Early blight",
    "Potato - Late blight", "Potato - healthy", "Raspberry - healthy", "Soybean - healthy",
    "Squash - Powdery mildew", "Strawberry - Leaf scorch", "Strawberry - healthy",
    "Tomato - Bacterial spot", "Tomato - Early blight", "Tomato - Late blight", "Tomato - Leaf Mold",
    "Tomato - Septoria leaf spot", "Tomato - Spider mites (Two-spotted spider mite)",
    "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic virus",
    "Tomato - healthy",
]


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    names = [layer.name for layer in model.layers]
    # Detect which of the 3 models was saved, so we use the same preprocessing as in training
    if "conv5_block3_out" in names:
        return model, "ResNet50", tf.keras.applications.resnet50.preprocess_input, "conv5_block3_out"
    if "out_relu" in names:
        return model, "MobileNetV2", tf.keras.applications.mobilenet_v2.preprocess_input, "out_relu"
    return model, "Custom CNN", lambda x: x / 255.0, "last_conv"


def grad_cam(model, x, layer_name, cls):
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        score = preds[0, cls]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], (IMG, IMG))[..., 0].numpy()


st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")
st.title("🌿 Plant Disease Detection")
st.write("Upload a photo of a single leaf. The model identifies the crop and disease "
         "(38 classes, 14 crops) and shows where it looked using Grad-CAM.")

model, model_name, preprocess, last_conv = load_model()

with st.sidebar:
    st.header("About")
    st.write(f"**Model:** {model_name} (transfer learning)")
    st.write("**Dataset:** PlantVillage, 54,303 images, 38 classes")
    st.write("**Compared:** Custom CNN, MobileNetV2, ResNet50")
    st.write("**Explainability:** Grad-CAM")

file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if file:
    img = np.array(Image.open(file).convert("RGB").resize((IMG, IMG))).astype("float32")
    x = preprocess(img.copy())[None, ...]
    probs = model.predict(x, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]
    best = int(top3[0])

    crop, disease = CLASS_NAMES[best].split(" - ", 1)
    if "healthy" in disease:
        st.success(f"**{crop}: Healthy leaf** ({probs[best]*100:.1f}% confidence)")
    else:
        st.error(f"**{crop}: {disease}** ({probs[best]*100:.1f}% confidence)")

    cam = grad_cam(model, tf.convert_to_tensor(x), last_conv, best)
    heat = cm.jet(cam)[..., :3]
    overlay = 0.6 * (img / 255.0) + 0.4 * heat

    big = lambda a: Image.fromarray((np.clip(a, 0, 1) * 255).astype("uint8")).resize((320, 320))
    col1, col2, col3 = st.columns(3)
    col1.image(big(img / 255.0), caption="Uploaded leaf")
    col2.image(big(overlay), caption="Grad-CAM (red = where the model looked)")
    with col3:
        st.write("**Top 3 predictions**")
        chart = pd.DataFrame({"Confidence (%)": [probs[i] * 100 for i in top3]},
                             index=[CLASS_NAMES[i] for i in top3])
        st.bar_chart(chart, horizontal=True)
else:
    st.info("Upload a leaf image to get a diagnosis.")
