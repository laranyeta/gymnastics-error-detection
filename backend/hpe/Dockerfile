FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

RUN apt-get update && apt-get install -y git ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY sapiens ./sapiens

WORKDIR /workspace/sapiens/pose

RUN pip install --no-cache-dir -U openmim

RUN mim install "mmpretrain>=1.0.0"

RUN pip install --no-cache-dir scikit-learn "numpy<2.0.0"

RUN pip install --no-cache-dir -r requirements/build.txt \
    -r requirements/runtime.txt \
    -r requirements/tests.txt \
    -r requirements/optional.txt

CMD ["/bin/bash"]