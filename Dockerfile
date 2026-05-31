FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ core/
COPY engines/ engines/
COPY utils/ utils/
COPY ui/ ui/
COPY config.py app.py ./

EXPOSE 7861

CMD ["streamlit", "run", "ui/app.py", "--server.port=7861", "--server.address=0.0.0.0", "--server.headless=true"]
