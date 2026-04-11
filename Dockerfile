FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY prompts/ ./prompts/
COPY main.py .

RUN mkdir -p data/raw data/processed data/external

ENV PYTHONPATH=/app/src

CMD ["python", "main.py"]
