@echo off

call .dashenv\Scripts\activate

start cmd /k "streamlit run notebooks_scripts\enron_streamlit.py --server.port 8080"

pause
