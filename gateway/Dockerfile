FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VLLM_URL=http://localhost:8000 \
    BASE_MODEL=Qwen/Qwen2.5-7B-Instruct

WORKDIR /app

COPY gateway/requirements.txt /app/gateway/requirements.txt
RUN pip install --no-cache-dir -r /app/gateway/requirements.txt

COPY gateway /app/gateway

EXPOSE 9000

CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "9000"]
