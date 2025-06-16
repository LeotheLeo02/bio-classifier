FROM python:3.11-slim
WORKDIR /app
COPY classify_api/ classify_api/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "classify_api.app:app", "--host", "0.0.0.0", "--port", "8000"]