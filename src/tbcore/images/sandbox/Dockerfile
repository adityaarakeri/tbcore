# Base image pinned by digest for running untrusted/generated content.
# python:3.11-slim index digest from Docker Hub on 2026-10-13; refresh deliberately.
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce

RUN useradd --create-home --uid 1000 sandbox \
    && mkdir -p /work \
    && chown -R sandbox:sandbox /work
USER sandbox
WORKDIR /work
