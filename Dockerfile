# Python version can be changed, e.g.
FROM python:3.10
# FROM ghcr.io/mamba-org/micromamba:1.5.1-focal-cuda-11.3.1
#FROM docker.io/python:3.12.1-slim-bookworm

LABEL org.opencontainers.image.authors="FNNDSC <dev@babyMRI.org>" \
      org.opencontainers.image.title="My ChRIS Plugin to detect text in DICOMs" \
      org.opencontainers.image.description="A ChRIS plugin to detect PHI containing text in a DICOMs"

ARG SRCDIR=/usr/local/src/pl-phi_detector
WORKDIR ${SRCDIR}

COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 tesseract-ocr  -y
# Copy your downloader script into the container
COPY nltk_downloader.py .

# Download NLTK resources
RUN python nltk_downloader.py


COPY . .
ARG extras_require=none
RUN pip install ".[${extras_require}]" \
    && cd / && rm -rf ${SRCDIR}
WORKDIR /

CMD ["phi_detector"]
