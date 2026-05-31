# Прототип балансировщика нагрузки ServerBalancer
FROM python:3.12-slim

# Системные зависимости для решателя CBC (PuLP) и сборки колёс.
RUN apt-get update && apt-get install -y --no-install-recommends \
        coinor-cbc curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — кэшируется отдельным слоем.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем исходный код.
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Health-check обращается к /health REST-сервиса.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
