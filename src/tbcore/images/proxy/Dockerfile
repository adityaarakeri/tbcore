# Base image pinned by digest.
# python:3.11-slim index digest from Docker Hub on 2026-10-13; refresh deliberately.
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce

# Both listen ports are above 1024, so the proxy needs no privileges. Debian ships an
# unprivileged `proxy` user (uid 13) with no login shell.
COPY proxy.py /usr/local/bin/tbcore-proxy.py
USER proxy
EXPOSE 3128 8080
CMD ["python", "/usr/local/bin/tbcore-proxy.py"]
