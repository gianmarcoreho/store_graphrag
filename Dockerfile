FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (data/ + store/ package).
COPY . .

EXPOSE 8501

# Default command = Phase 2 (Streamlit UI).
# Phase 1 (graph build) is run as:  python -m store.build_graph
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
