# The index digest keeps this base image pinned across supported architectures.
# node:22-bookworm-slim index digest from Docker Hub on 2026-10-14.
FROM node:22-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392

ARG CODEX_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && test -n "$CODEX_VERSION" \
    && npm install --global "@openai/codex@${CODEX_VERSION}" \
    && mkdir -p /work \
    && chown node:node /work

USER node
WORKDIR /work
ENV HOME=/work
ENV CODEX_HOME=/work/.codex
