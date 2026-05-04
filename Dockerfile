FROM python:3.13-slim

WORKDIR /app

# System deps: curl (healthcheck), build-essential (some Python packages need gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid pulling the 2.5 GB CUDA build
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.headless", "true", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0"]
