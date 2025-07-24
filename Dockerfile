FROM python:3.11-slim

WORKDIR /app

# install deps first, so Docker layer caching works
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your FastAPI code (render runs from within classify_api/)
COPY app/ app/

EXPOSE 8000

# module path: app/app.py → module "app.app", app instance "app"
ENTRYPOINT ["sh","-c","uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8080}"]