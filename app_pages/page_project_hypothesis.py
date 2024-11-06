import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    """
    Renders the Project Hypothesis and Validation section in the Streamlit app.
    
    This function presents the project's hypothesis regarding the distinguishable 
    patterns in brain MRI images for different tumor types and outlines the 
    methodologies used to validate this hypothesis.
    """
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"**1. Visual Differentiation**\n"
        f"* Hypothesis: MRI scans of healthy brains and those affected by different tumor types (Glioma, Meningioma, Pituitary) exhibit unique visual patterns distinguishable by machine learning models.\n"
        f"* Validation: Successfully validated through the model’s performance in test accuracy (96.31%) and low test loss (0.11078). Additionally, confusion matrix analysis confirmed that each class demonstrated distinct, recognizable visual features, leading to high classification accuracy across the four categories.\n\n"

        f"**2. Multi-class Classification**\n"
        f"* Hypothesis: A deep learning model, specifically one using a Convolutional Neural Network (CNN) architecture, can accurately classify MRI brain scans into four categories (Glioma, Meningioma, Pituitary tumor, and No Tumor) based on distinctive features of each type.\n"
        f"* Validation: Fully validated by the model’s high test accuracy (96.31%), indicating strong classification capabilities. Performance metrics like precision, recall, and F1-score for each class were within acceptable ranges, confirming the model's ability to reliably differentiate among the four categories in real-world MRI scan analyses.\n\n"
    
        f"**3. Data Augmentation**\n"
        f"* Hypothesis: Data augmentation techniques, such as rotation, flipping, and brightness adjustment, would improve the model's generalization and help mitigate overfitting by providing diverse examples of MRI images.\n"
        f"* Validation: This hypothesis was validated as the data augmentation contributed significantly to the model's stable and high performance. The test accuracy of 96.31% without signs of overfitting in the accuracy and loss curves indicates effective generalization, supporting the utility of augmentation in enhancing model robustness.\n"
    
    )

    st.info(
        f"The brain tumor classification model achieved strong results with a test accuracy of 96.31% and a test loss of 0.11078, but it did not reach 100% accuracy. The model's performance was limited by the diversity and quantity of the dataset, as well as variations in real-world MRI scans that differ from the dataset's structure. Improvements could come from increasing the dataset size to capture more variability, adding MRI images from different views (such as sagittal and coronal views), and further tuning hyperparameters. Training for additional epochs with robust data augmentation may also enhance generalization to new, unseen MRI scans."
    )