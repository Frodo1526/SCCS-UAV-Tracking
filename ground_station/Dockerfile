FROM ubuntu:24.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY common/ /app/common/
COPY ground_station/ /app/ground_station/

# Overrides confirmation error and removes cached packages
RUN pip3 install --break-system-packages --no-cache-dir \
    -r /app/ground_station/requirements-docker.txt

ENV PYTHONPATH=/app
WORKDIR /app/ground_station

ENTRYPOINT ["python3", "main.py"]