import numpy as np
import pandas as pd
import os
import base64
from datetime import datetime
import joblib


def download_dataframe_as_csv(df):
    """
    Convert a DataFrame to a CSV file and create a download link.

    Parameters:
    - df: The DataFrame to be converted to CSV.

    Returns:
    - href: A string containing the HTML anchor tag to download the CSV file.
    """
    datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" download="Report {datetime_now}.csv" '
        f'target="_blank">Download Report</a>'
    )
    return href


def load_pkl_file(file_path):
    """
    Load a pickle file and return its contents.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - The contents of the loaded pickle file.
    """
    return joblib.load(filename=file_path)
