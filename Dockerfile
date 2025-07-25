
# CUDA 12.1과 Python 3.10 기반 이미지 사용 (베이스 이미지와 일치)
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 환경 변수 설정 (필수만)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# apt 임시 디렉토리 권한 문제 해결 및 시스템 패키지 설치
RUN mkdir -p /tmp && chmod 1777 /tmp && \
    apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 작업 디렉토리 설정
WORKDIR /workspace

# 호환되는 패키지 버전으로 설치 (의존성 충돌 방지)
RUN pip install \
    transformers==4.40.0 \
    datasets==2.19.0 \
    accelerate==0.30.0 \
    timm==0.9.16 \
    wandb==0.17.0 \
    Pillow==11.1.0 \
    opencv-python==4.11.0.86 \
    scipy==1.15.1 \
    scikit-learn==1.6.1 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    grad-cam==1.5.4 \
    huggingface-hub==0.23.0 \
    pandas==2.2.3 \
    ipykernel==6.29.5 \
    jupyter_client==8.6.3



# 포트 설정 (Jupyter, TensorBoard 등을 위해)
EXPOSE 8888 6006

# 기본 명령어 설정
CMD ["/bin/bash"]