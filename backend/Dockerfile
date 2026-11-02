FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/*requirements.txt backend/
COPY RTDETR-main-code/requirements.txt RTDETR-main-code/
RUN pip install --no-cache-dir "setuptools<81" wheel && pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r backend/model-requirements.txt -r backend/requirements.txt
COPY RTDETR-main-code/ultralytics RTDETR-main-code/ultralytics
COPY backend/app backend/app
CMD ["python","-m","uvicorn","backend.app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
