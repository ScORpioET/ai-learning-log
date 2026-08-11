FROM python:3.10-slim

WORKDIR /app

# git 是安裝 transformers(從 GitHub commit 裝)需要用到的指令,slim 版預設沒裝
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf
    
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_gpt2.py .
COPY config/ ./config/
COPY input.txt .

CMD ["python", "train_gpt2.py"]