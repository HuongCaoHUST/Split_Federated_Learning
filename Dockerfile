FROM ultralytics/ultralytics:latest

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://get.docker.com | sh \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    -r /tmp/requirements.txt \
    requests==2.32.3 \
    prometheus_client \
    pandas

CMD ["bash"]
