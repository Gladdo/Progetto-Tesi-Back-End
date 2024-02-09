FROM ubuntu:22.04

RUN apt-get update
RUN apt install --assume-yes python3.10 python3.10-venv pip
RUN apt install --assume-yes git

RUN pip install --no-cache-dir django djangorestframework django-cors-headers pillow
RUN pip install --no-cache-dir torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir controlnet_aux==0.0.7
RUN pip install --no-cache-dir diffusers==0.21.4 transformers==4.34.1 accelerate==0.23.0
RUN pip install --no-cache-dir xformers==0.0.23.post1

RUN pip install sentence-transformers tzdata

COPY djangoproject app/djangoproject

# Training App:
COPY training-app-slim app/training-app
COPY requirements-slim.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get install ffmpeg libsm6 libxext6  -y
