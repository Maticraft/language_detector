FROM python:3.11-slim

SHELL ["/bin/bash", "-c"]
WORKDIR /app

COPY requirements.txt .

RUN apt-get update &&\
    apt-get install --no-install-recommends --yes build-essential
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
