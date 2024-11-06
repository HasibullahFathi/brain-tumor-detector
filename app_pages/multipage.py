import streamlit as st


# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    """
    A class to manage multiple pages in a Streamlit app.

    Attributes:
        app_name (str): The name of the app.
        pages (list): A list of pages added to the app.
    """

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🖥️")

    def add_page(self, title, func) -> None:
        """Add a page to the app."""
        self.pages.append({"title": title, "function": func})

    def run(self):
        """Run the app, displaying the selected page."""
        st.title(self.app_name)
        page = st.sidebar.radio(
            'Menu', self.pages, format_func=lambda page: page['title'])
        page['function']()
