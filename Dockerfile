FROM ultralytics/ultralytics:latest

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://get.docker.com | sh \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    requests==2.32.3 \
    pika==1.3.2\
    mlflow\
    prometheus_client

CMD ["bash"]
