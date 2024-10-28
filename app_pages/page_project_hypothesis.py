import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"This project hypothesizes that brain MRI images with Glioma, Meningioma, or Pituitary tumors "
        f"exhibit distinguishable patterns or characteristics that differentiate them from normal (non-tumor) images." 
        f"We expect that variations in texture, intensity, and structural patterns specific to each tumor type will aid" 
        f"in the accurate classification of MRI images.\n\n"
        f"To validate this hypothesis, we perform a series of studies, including:\n"
        f"* Image Montage Analysis: Initial montages of the MRI images reveal potential differences across tumor types." 
        f"Glioma, Meningioma, and Pituitary tumor images exhibit varying shapes and densities in affected areas that may aid in classification.\n"
        f"* Average and Variability Image Analysis: The average and variability images for each class were generated to detect common patterns or textures." 
        f"However, these studies have not yielded consistent, visually discernible patterns that reliably distinguish one class from another.\n"
        f"* Difference between Averages: A comparison of average images between tumor types and the non-tumor category provides some subtle distinctions in intensity." 
        f"However, these differences alone are insufficient for a clear, human-detectable pattern for differentiation, suggesting that deeper learning models may be required to uncover meaningful features.\n\n"
        f"These findings support the hypothesis that machine learning models, particularly convolutional neural networks," 
        f"may effectively classify brain MRI images by leveraging subtle patterns across tumor types that are challenging to identify through visual inspection alone."

    )
