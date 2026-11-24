"""Navigation MaintiDoc : questions documentaires et espace tickets."""
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

UI_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UI_ROOT.parents[1]
load_dotenv(PROJECT_ROOT / ".env")
st.set_page_config(page_title="MaintiDoc | Assistant SAN", page_icon="???", layout="wide")

pages = {
    "MaintiDoc": [
        st.Page(str(UI_ROOT / "app_pages" / "questions.py"), title="Manuels et questions", icon=":material/menu_book:", default=True),
        st.Page(str(UI_ROOT / "app_pages" / "tickets.py"), title="Tickets", icon=":material/confirmation_number:"),
    ]
}
st.navigation(pages, position="sidebar").run()
