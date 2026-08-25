# Pinned base image for running untrusted/generated content.
FROM python:3.11-slim

RUN useradd --create-home --uid 1000 sandbox \
    && mkdir -p /work \
    && chown -R sandbox:sandbox /work
USER sandbox
WORKDIR /work
