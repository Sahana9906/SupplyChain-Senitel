$env:PYTHONPATH="src"
python -m logisense.cli db-upgrade
#Backend COMMAND 
$env:PYTHONPATH="src"
uvicorn logisense.api:app --host 127.0.0.1 --port 8000
#FRONTEND
cd .\frontend
python -m http.server 8080
