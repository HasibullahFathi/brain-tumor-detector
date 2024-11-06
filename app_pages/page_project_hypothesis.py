import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    """
    Renders the Project Hypothesis and Validation section in the Streamlit app.
    This function presents project's hypothesis regarding the distinguishable
    patterns in brain MRI images for different tumor types and outlines the
    methodologies used to validate this hypothesis.
    """
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"**1. Visual Differentiation**\n"
        f"* Hypothesis: MRI scans of healthy brains and those affected by "
        f"different tumor types (Glioma, Meningioma, Pituitary) exhibit "
        f"unique visual patterns distinguishable by machine learning models.\n"
        f"* Validation: Successfully validated through the model’s performance"
        f" in test accuracy (96.31%) and test loss (0.11078). Additionally, "
        f"confusion matrix analysis confirmed that each class demonstrated "
        f"distinct, recognizable visual features, leading to high "
        f"classification accuracy across the four categories.\n\n"
    )

    st.success(
        f"**2. Multi-class Classification**\n"
        f"* Hypothesis: A deep learning model, specifically one using a "
        f"Convolutional Neural Network (CNN) architecture, can accurately "
        f" classify MRI brain scans into four categories (Glioma, Meningioma, "
        f"Pituitary tumor, and No Tumor) based on distinctive features"
        f"of each type.\n"
        f"* Validation: Fully validated by the model’s high test accuracy "
        f"(96.31%), indicating strong classification capabilities. Performance"
        f" metrics like precision, recall, and F1-score for each class were "
        f"within acceptable ranges, confirming "
        f"the model's ability to reliably differentiate among the four "
        f"categories in real-world MRI scan analyses.\n\n"
    )

    st.success(
        f"**3. Data Augmentation**\n"
        f"* Hypothesis: Data augmentation techniques, such as rotation, "
        f"flipping, and brightness adjustment, would improve the model's "
        f"generalization and help mitigate overfitting by providing diverse "
        f"examples of MRI images.\n"
        f"* Validation: This hypothesis was validated as the data augmentation"
        f" contributed significantly to the model's high performance."
        f"The test accuracy of 96.31% without signs of overfitting in the "
        f"accuracy and loss curves indicates effective generalization, "
        f"supporting the utility of augmentation in enhancing model "
        f"robustness.\n"
    )

    st.info(
        f"The brain tumor classification model achieved strong results with"
        f"a test accuracy of 96.31% and a test loss of 0.11078, but it did not"
        f"reach 100% accuracy. "
        f"The model's performance was limited by the diversity and quantity of"
        f" the dataset, as well as variations in real-world MRI scans that "
        f"differ from the dataset's structure."
        f"Improvements could come from increasing the dataset size to capture "
        f"more variability, adding MRI images from different views (such as "
        f"sagittal and coronal views), and further tuning hyperparameters."
        f"Training for additional epochs with robust data augmentation "
        f"may also enhance generalization to new, unseen MRI scans."
    )
    