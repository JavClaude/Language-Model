import subprocess


def main() -> None:
    subprocess.call(
        ["streamlit", "run", "language_modeling/infra/application/frontend/frontend.py"]
    )
