@echo off
echo ==========================================
echo Used Car Price Prediction
echo ==========================================
echo.
echo Installing requirements...
pip install -r requirements.txt
echo.
echo Training model...
python train_model.py
echo.
echo Starting Streamlit application...
streamlit run app.py
pause
