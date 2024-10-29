import streamlit as st
import matplotlib.pyplot as plt

def page_summary_body():
    """
    Renders a summary of the project, detailing the importance of brain tumor classification,
    the dataset used, and the project requirements.
    """
    st.write("### Quick Project Summary")

    st.info(
        f"**What is a Brain Tumor?**\n\n"
        
        f"A brain tumor is an abnormal mass or cluster of cells within the brain. Because the brain is enclosed by the skull—a rigid, " 
        f"confined space—any growth within this area can lead to serious complications. Brain tumors can be classified as either" 
        f"malignant (cancerous) or benign (noncancerous). As these tumors enlarge, they can increase pressure within the skull, " 
        f"which may result in brain damage and become life-threatening.\n\n"

        f"**The Importance of Brain Tumor Classification**\n\n"

        f"Early detection and precise classification of brain tumors are essential areas of research in medical imaging." 
        f"Effective classification helps identify the most appropriate treatment options and can significantly improve patient outcomes, potentially saving lives.\n\n"

        f"**Project Dataset**\n\n"

        f"The dataset used in this project contains a total of 7023 MRI images across four categories: Glioma, Meningioma, Pituitary tumor, and No Tumor (normal brain scans)." 
        f"This dataset has been organized for training and testing purposes and provides a comprehensive base for building a machine learning model capable of differentiating between these conditions. "
        f"Each MRI image represents a slice of the brain in grayscale format, capturing variations in structure and tissue density associated with each tumor type, which can be instrumental in automated detection and classification."
    )

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/HasibullahFathi/brain-tumor-detector/blob/main/README.md)."
    )

    st.success(
        f"The project has two main business requirements:\n"
        f"* 1 - The client is interested in a study that visually differentiates between "
        f"the types of brain tumors, including Glioma, Meningioma, Pituitary, and healthy brain tissues.\n"
        f"* 2 - The client needs a tool that can accurately identify whether a given brain MRI scan contains a tumor and, if so, classify its type."
    )

