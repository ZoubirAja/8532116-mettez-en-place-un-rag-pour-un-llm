FROM python:3.14-slim

WORKDIR /app

# Dépendances d'abord : cette étape ne se refait que si requirements.txt change (cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application (pas les données : vector_db/ et database/ sont montés en volume)
COPY utils/ ./utils/
COPY pages/ ./pages/
COPY api/ ./api/
COPY puls_events_app.py .

EXPOSE 8000

CMD ["uvicorn", "api.puls_events_api:app", "--host", "0.0.0.0", "--port", "8000"]
