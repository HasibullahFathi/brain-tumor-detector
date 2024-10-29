import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities
                                                    )

def page_brain_tumor_detector_body():
    """
    Renders the Brain Tumor Detector page in the Streamlit app.
    
    This function allows users to upload brain MRI images and receive 
    predictions on whether they exhibit signs of Glioma, Meningioma, or 
    Pituitary tumors. It also provides a link to download a dataset 
    of classified images for further analysis.
    """
    st.info(
        f"* The client is interested in conducting a study to visually differentiate" 
        f" between a normal brain and one with Glioma, Meningioma, or Pituitary tumors."
        )

    st.write(
        f"* You can download a set of classified brain tumor MRI images for live prediction. "
        f"You can download the images from [this Kaggle dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)."
        )

    st.write("---")

    images_buffer = st.file_uploader('Upload Brain MRI Image. You may select more than one.',
                                     type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    # Initialize `df_report` once outside the loop
    df_report = pd.DataFrame(columns=["Name", "Result"])

    if images_buffer is not None:
        for image in images_buffer:
            img_pil = Image.open(image)
            st.info(f"MRI image: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            # Add each prediction to `df_report` without reinitializing it
            df_report = pd.concat([df_report, pd.DataFrame([{"Name": image.name, "Result": pred_class}])],
                                  ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
