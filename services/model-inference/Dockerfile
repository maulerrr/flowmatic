FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY services/model-inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY models /app/models
COPY services/model-inference/src /app/src

ENV PYTHONPATH=/app/models:/app/src
ENV FLOWMATIC_ROOT=/app
ENV PORT=8093
ENV HF_HOME=/cache/huggingface
ENV CHECKPOINTS_DIR=/app/models/checkpoints
ENV INFER_DEVICE=auto

WORKDIR /app/src
EXPOSE 8093

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8093"]
