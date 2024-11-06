import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():
    """
    Renders a summary of the project, detailing the importance
    of brain tumor classification, the dataset used, and the
    project requirements.
    """
    st.write("### Quick Project Summary")

    st.info(
        f"**What is a Brain Tumor?**\n\n"
        f"A brain tumor is an abnormal mass or cluster of cells "
        f"within the brain. Because the brain is enclosed by the "
        f"skull—a rigid, confined space—any growth within this area "
        f"can lead to serious complications. Brain tumors can be "
        f"classified as either malignant (cancerous) or benign "
        f"(noncancerous). As these tumors enlarge, they can increase "
        f"pressure within the skull, which may result in brain damage "
        f"and become life-threatening.\n\n"

        f"**The Importance of Brain Tumor Classification**\n\n"

        f"Early detection and precise classification of brain tumors "
        f"are essential areas of research in medical imaging. "
        f"Effective classification helps identify the most appropriate "
        f"treatment options and can significantly improve patient "
        f"outcomes, potentially saving lives.\n\n"

        f"**Project Dataset**\n\n"

        f"The dataset used in this project contains a total of 7023 MRI "
        f"images across four categories: Glioma, Meningioma, Pituitary tumor, "
        f"and No Tumor (normal brain scans). This dataset has been organized "
        f"for training, testing and validation purposes and provides a "
        f"comprehensive base for building a machine learning model capable of "
        f"differentiating between these conditions. Each MRI image represents "
        f"a slice of the brain in grayscale format, capturing variations in "
        f"structure and tissue density associated with each tumor type, which "
        f"can be instrumental in automated detection and classification."
    )

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/HasibullahFathi/"
        f"brain-tumor-detector/blob/main/README.md)."
    )

    st.success(
        f"The project has two main business requirements:\n"
        f"* 1 - The client is interested in a study that visually "
        f"differentiates between the types of brain tumors, including Glioma, "
        f"Meningioma, Pituitary, and healthy brain tissues.\n"
        f"* 2 - The client needs a tool that can accurately identify "
        f"whether a given brain MRI scan contains a tumor and, if so, "
        f"classifyits type."
    )
