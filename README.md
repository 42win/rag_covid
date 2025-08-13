1. inisialisasi environment
```
python -m venv env
```

2. aktivasi environment
```
.\env\Scripts\activate
```

3. install library
```
pip install -r requirements.txt
```

4. running app
```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. prepare model for the first time
```
http://localhost:8000/prepare_model
```