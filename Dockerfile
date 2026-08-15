FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential make \
    && rm -rf /var/lib/apt/lists/*

# Pinned deps installed at image build; source is bind-mounted and used via PYTHONPATH.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    numpy==2.1.3 \
    scipy==1.14.1 \
    einops==0.8.1 \
    matplotlib==3.9.4 \
    pandas==2.2.3 \
    pyyaml==6.0.2 \
    tqdm==4.67.1 \
    pytest==8.3.4

ENV PYTHONPATH=/workspace/src \
    OMP_NUM_THREADS=3 \
    MKL_NUM_THREADS=3 \
    PYTHONDONTWRITEBYTECODE=1

# uid matches the host user so the bind mount is writable
RUN useradd -m -u 1001 -s /bin/bash dev

WORKDIR /workspace
USER dev
CMD ["sleep", "infinity"]
