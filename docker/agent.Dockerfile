# docker/agent.Dockerfile
#
# Supervisor(main.py, :8001) + A2A 서버 3종(app.a2a.server knowledge/execution/perception,
# :8002~8004) 공용 이미지. 4개 프로세스 모두 로컬 모델 서버(app/model_server)를
# ChatOpenAI(base_url=MODEL_SERVER_URL)로 HTTP 호출만 하고 자체적으로 GPU 텐서 연산을
# 하지 않으므로 CUDA 베이스 이미지가 필요 없다(docker/model-server.Dockerfile과 대비).
#
# 실행 대상은 docker-compose의 command:로 프로세스별로 갈라진다:
#   supervisor  → python main.py
#   knowledge   → python -m app.a2a.server knowledge
#   execution   → python -m app.a2a.server execution
#   perception  → python -m app.a2a.server perception
#
# 빌드 (repo 루트에서):
#   docker build -f docker/agent.Dockerfile -t driving-copilot-agent:latest .

FROM python:3.10-slim

WORKDIR /app

# sentence-transformers/langchain-huggingface가 의존성으로 torch를 끌고 오는데, 기본
# PyPI 인덱스는 CUDA 번들 wheel(수 GB)을 줄 수 있다 — 이 이미지는 CPU 전용이므로
# 먼저 CPU 전용 wheel을 명시적으로 깔아 GPU wheel이 받아지는 것을 막는다.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY docker/requirements-agent.txt ./requirements-agent.txt
RUN pip install --no-cache-dir -r requirements-agent.txt

COPY app ./app
COPY main.py ./main.py

# Supervisor(8001) + A2A 서버 3종(8002~8004). 실제 어느 포트를 여는지는 command:로
# 실행되는 프로세스에 따라 다르므로 4개 전부 문서화 목적으로 선언한다.
EXPOSE 8001 8002 8003 8004

RUN useradd --create-home --uid 1000 appuser
USER appuser

# 기본값은 Supervisor. A2A 서버 3종은 docker-compose의 command:로 override한다.
CMD ["python", "main.py"]
