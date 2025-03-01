FROM python:3.10.16-slim

WORKDIR /product-matching-app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN mkdir .streamlit
COPY .streamlit/secrets.toml .streamlit/secrets.toml

COPY product_matching_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "/product-matching-app/product_matching_app.py", "--server.port=8501", "--server.address=0.0.0.0"]