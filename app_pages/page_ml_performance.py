import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    """
    Renders the Machine Learning Performance Metrics page in the Streamlit app.
    This function displays the frequency distribution of labels in the train,
    validation, and test sets, model training history (accuracy and losses),
    evaluation results on the test set, confusion matrix, and model
    visualization.
    """
    version = 'v1'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    labels_distribution = plt.imread(
        f"outputs/{version}/labels_distribution.png")
    st.image(
        labels_distribution, 
        caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")


    st.write("### Model History")
    col1, col2 = st.columns(2)
    with col1: 
        model_acc = plt.imread(
            f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(
            f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(
        pd.DataFrame(
            load_test_evaluation(version), 
            index=['Loss', 'Accuracy']))
    st.write("---")

    st.write("### Confusion Matrix")
    confusion_matrix = plt.imread(
        f"outputs/{version}/confusion_matrix.png")
    st.image(confusion_matrix, caption='Confusion Matrix')
    st.write("---")

    st.write("### Model Visualization")
    model_visualization = plt.imread(
        f"outputs/{version}/model_visualization.png")
    st.image(model_visualization)
    st.write("---")