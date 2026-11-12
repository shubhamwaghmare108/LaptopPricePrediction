FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY laptop_price ./laptop_price
COPY backend ./backend
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
