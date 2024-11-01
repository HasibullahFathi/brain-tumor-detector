import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow import keras
from PIL import Image
from src.data_management import load_pkl_file

# Define the class labels
target_map = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results for each class.
    
    Parameters:
    - pred_proba: Prediction probabilities for each class
    """

    # Create a DataFrame to hold class probabilities
    prob_per_class = pd.DataFrame(
        data=pred_proba,
        index=target_map.values(),
        columns=['Probability']
    ).round(3)
    prob_per_class['Diagnostic'] = prob_per_class.index

    # Plotting the class probabilities
    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        range_y=[0, 1],
        width=600, height=300,
        template='seaborn'
    )
    st.plotly_chart(fig)


def resize_input_image(img, version):
    """
    Reshape image to average image size and add color channels.
    """
    image_shape = load_pkl_file(file_path=f"outputs/v1/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)

    # Convert to RGB if not already (ensures 3 channels)
    img_rgb = img_resized.convert("RGB")

    # Expand dimensions to match model input shape and normalize
    my_image = np.expand_dims(img_rgb, axis=0) / 255.0

    return my_image



def load_model_and_predict(my_image, version):
    """
    Load the model and perform a prediction on the input image.
    
    Parameters:
    - my_image: Preprocessed image for prediction
    - version: Model version to load
    
    Returns:
    - pred_proba: Array of probabilities for each class
    - pred_class: Predicted class label
    """
    
    # Load the model
    model = keras.models.load_model(f"outputs/v1/brain_tumor_detector.keras")


    # Predict probabilities for each class
    pred_proba = model.predict(my_image)[0]  # Model outputs an array of probabilities
    pred_class_idx = np.argmax(pred_proba)  # Get index of highest probability
    pred_class = target_map[pred_class_idx]  # Map to class name

    # Display the prediction result
    st.write(
        f"The predictive analysis indicates that the brain MRI is most likely "
        f"**{pred_class.lower()}**."
    )

    return pred_proba, pred_class
