@echo off
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Installation complete!
echo.
echo To run the app, use: streamlit run app.py
pause
