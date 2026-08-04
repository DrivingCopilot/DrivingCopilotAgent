# docker/model-server.Dockerfile
#
# 로컬 HuggingFace 모델 서버(app/model_server, :11500) 전용 GPU 이미지.
# Qwen2-VL-7B-Instruct(bitsandbytes 4bit 즉석 양자화) + Qwen2.5-1.5B-Instruct를
# transformers로 직접 로드해 OpenAI 호환 API로 서빙한다 — Supervisor/Knowledge/
# Execution/Perception(docker/agent.Dockerfile, CPU-only) 4개 프로세스가 이 서버
# 하나를 공유한다.
#
# 베이스로 nvidia/cuda 원본 대신 pytorch/pytorch 공식 이미지를 쓰는 이유: torch와
# 시스템 CUDA/cuDNN 버전이 이미 맞춰서 빌드돼 있어, 버전 궁합을 직접 맞출 필요가 없다.
# CUDA 12.4는 A6000(Ampere, compute capability 8.6)을 문제없이 지원한다.
#
# 빌드 (repo 루트에서):
#   docker build -f docker/model-server.Dockerfile -t driving-copilot-model-server:latest .
# 실행 시 반드시 GPU를 붙여야 한다:
#   docker run --gpus all -p 11500:11500 driving-copilot-model-server:latest
#
# 최초 기동 시 HuggingFace에서 7B(비양자화, ~15GB) + 1.5B 가중치를 내려받는다
# (README 참고) — 컨테이너 재생성 때마다 다시 받지 않도록 반드시 아래처럼
# ~/.cache/huggingface를 named volume으로 마운트할 것 (docker-compose.yml에서 처리).

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# torch/torchvision/torchaudio는 베이스 이미지에 CUDA 빌드로 이미 포함돼 있으므로
# requirements-model-server.txt에는 torch를 넣지 않는다(재설치 방지).
COPY docker/requirements-model-server.txt ./requirements-model-server.txt
RUN pip install --no-cache-dir -r requirements-model-server.txt

COPY app ./app

EXPOSE 11500

# 모델 가중치 다운로드 캐시. docker-compose에서 named volume으로 마운트해
# 컨테이너 재생성 시에도 재다운로드하지 않도록 한다.
ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "-m", "app.model_server.server"]
