FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

RUN apt-get update && apt-get install -y git ffmpeg libgl1

WORKDIR /workspace
COPY sapiens ./sapiens

WORKDIR /workspace/sapiens
RUN pip install --no-cache-dir -r pose/requirements.txt

CMD ["/bin/bash"]
