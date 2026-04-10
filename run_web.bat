@echo off
echo ================================
echo Astra ML System - Web Interface
echo ================================

streamlit run web_app.py --server.address localhost --server.port 8501

pause