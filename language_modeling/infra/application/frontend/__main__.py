import os

import subprocess

STREAMLIT_SERVER_PORT = os.environ.get("STREAMLIT_SERVER_PORT")

def main() -> None:
    subprocess.call(
        ["streamlit", "run", "language_modeling/infra/application/frontend/frontend.py"]
    )
