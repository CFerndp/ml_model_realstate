rem pip install -r requirements.txt
rem python -m app.train_model
uvicorn app.main:app --reload
