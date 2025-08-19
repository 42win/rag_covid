1. inisialisasi environment
```
python -m venv env
```

2. aktivasi environment
```
source venv/bin/activate
```

3. install library
```
pip install -r requirements.txt
```

4. lengkapi token hugging face anda di file .env
```
.env
```

5. running app
```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

6. prepare model for the first time
```
http://localhost:8000/prepare_model
```

7.  build knowledgebase
```
http://localhost:8000/build-kb
```

8.  testing
```
http://localhost:8000/rag-answer
```
contoh question
```
apa tujuan Rencana Operasi COVID-19?
apa yang dimakud S-Gene Target Failure?
kapan pertama kali penyakit covid menyebar di indonesia?

sebutkan ruang lingkup surat edaran satgas terkait protokol perjalanan luar negeri
apa maksud dan tujuan dari surat edaran satgas terkait protokol perjalanan luar negeri?
apa kriteri wni yang dapat memasuki wilayah indonesia ?

sebutkan dasar hukum rencana operasi penanggulangan dan pencegahan covid 19 ?
berapa jumlah penduduk indonesia pada tahun 2018 ?
daerah mana yang memiliki mobilitas tertinggi?
```

