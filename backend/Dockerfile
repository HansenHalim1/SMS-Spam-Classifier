FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY artifacts /app/artifacts
COPY backend /app/backend

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV ALLOWED_ORIGINS="http://localhost:8501,http://localhost:3000,https://*.hf.space"

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
