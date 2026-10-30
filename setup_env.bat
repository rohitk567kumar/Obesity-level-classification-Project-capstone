@echo off
REM Create a virtual environment and install the libraries (Windows).
REM Run:  setup_env.bat

python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name obesity-env --display-name "Python (obesity-env)"

echo.
echo Done. Activate the environment with:  .venv\Scripts\activate
