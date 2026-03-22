FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY data/ data/
COPY models/ models/
COPY src/ src/

# 🔥 TRAIN MODEL DURING BUILD
RUN python data/generate_dataset.py && \
    python src/preprocess.py && \
    python src/train_model.py

EXPOSE $PORT

CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:$PORT backend.app:app"]