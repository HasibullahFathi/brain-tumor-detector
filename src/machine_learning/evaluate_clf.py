import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation(version):
    """
    Load test evaluation data from a pickle file.

    Parameters:
    - version: The version identifier used to locate the evaluation file.

    Returns:
    - The contents of the evaluation data loaded from the pickle file.
    """
    return load_pkl_file(f'outputs/{version}/evaluation.pkl')
