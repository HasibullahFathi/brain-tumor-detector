import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random


def page_brain_tumor_visualizer_body():
    """
    Renders the Brain Tumor Visualizer page in the Streamlit app.
    
    This function includes options for:
    - Displaying average and variability images for each tumor type.
    - Comparing 'No Tumor' images against each tumor type.
    - Creating a montage of images for a selected tumor type.
    """
    st.write("### Brain Tumor Visualizer")
    st.info(
        f"* The client is interested in having a study that visually "
        f"differentiates among different types of brain tumors."
    )
    
    version = 'v1'
    if st.checkbox("Difference between average and variability image for each tumor type"):
      
        avg_glioma = plt.imread(f"outputs/{version}/avg_var_glioma.png")
        avg_meningioma = plt.imread(f"outputs/{version}/avg_var_meningioma.png")
        avg_pituitary = plt.imread(f"outputs/{version}/avg_var_pituitary.png")
        avg_no_tumor = plt.imread(f"outputs/{version}/avg_var_notumor.png")

        st.warning(
            f"* The average and variability images show subtle patterns "
            f"that may assist in differentiating tumor types. Some color and texture "
            f"variations are visible across categories."
        )

        st.image(avg_glioma, caption='Glioma - Average and Variability')
        st.image(avg_meningioma, caption='Meningioma - Average and Variability')
        st.image(avg_pituitary, caption='Pituitary - Average and Variability')
        st.image(avg_no_tumor, caption='No Tumor - Average and Variability')
        st.write("---")

    if st.checkbox("Differences between 'No Tumor' and each tumor type"):
      # Load each comparison image
      diff_notumor_vs_glioma = plt.imread(f"outputs/{version}/avg_diff_notumor_vs_glioma.png")
      diff_notumor_vs_meningioma = plt.imread(f"outputs/{version}/avg_diff_notumor_vs_meningioma.png")
      diff_notumor_vs_pituitary = plt.imread(f"outputs/{version}/avg_diff_notumor_vs_pituitary.png")

      # Display each comparison image with a caption
      st.warning(
          f"* The images below highlight differences between the 'No Tumor' average image "
          f"and each specific tumor type (Glioma, Meningioma, and Pituitary). Some texture "
          f"or intensity variations might be observed between the categories."
      )
      
      st.image(diff_notumor_vs_glioma, caption="Difference between 'No Tumor' and 'Glioma'")
      st.image(diff_notumor_vs_meningioma, caption="Difference between 'No Tumor' and 'Meningioma'")
      st.image(diff_notumor_vs_pituitary, caption="Difference between 'No Tumor' and 'Pituitary'")


    if st.checkbox("Image Montage"): 
        st.write("* To refresh the montage, click on the 'Create Montage' button")
        my_data_dir = 'inputs/brain-tumor-mri-dataset/mri-images'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(label="Select label", options=labels, index=0)
        
        if st.button("Create Montage"):      
            image_montage(dir_path=my_data_dir + '/validation',
                          label_to_display=label_to_display,
                          nrows=8, ncols=3, figsize=(10, 25))
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
    """
    Creates an image montage for the specified label from the directory.

    Args:
        dir_path (str): Path to the directory containing images.
        label_to_display (str): The label for which to create the montage.
        nrows (int): Number of rows in the montage.
        ncols (int): Number of columns in the montage.
        figsize (tuple): Size of the figure to be displayed.
    """
    sns.set_style("white")
    labels = os.listdir(dir_path)

    # Subset the class to display
    if label_to_display in labels:
        # Select a subset of images for the montage if too many images
        images_list = os.listdir(f"{dir_path}/{label_to_display}")
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.warning(
                f"Decrease nrows or ncols to create your montage. \n"
                f"There are only {len(images_list)} images in the subset. "
                f"You requested a montage with {nrows * ncols} spaces."
            )
            return

        # Setup plot grid and indices
        list_rows = range(nrows)
        list_cols = range(ncols)
        plot_idx = list(itertools.product(list_rows, list_cols))

        # Create Figure and display images
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for idx, ax_idx in enumerate(plot_idx):
            img = imread(f"{dir_path}/{label_to_display}/{img_idx[idx]}")
            img_shape = img.shape
            axes[ax_idx[0], ax_idx[1]].imshow(img)
            axes[ax_idx[0], ax_idx[1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
            axes[ax_idx[0], ax_idx[1]].set_xticks([])
            axes[ax_idx[0], ax_idx[1]].set_yticks([])
        
        plt.tight_layout()
        st.pyplot(fig=fig)

    else:
        st.error("The selected label does not exist.")
        st.write(f"Available options are: {labels}")
