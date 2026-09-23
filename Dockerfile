FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN adduser --disabled-password --no-create-home app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache-dir

COPY --chown=app:app *.py ./

# Fail the build on a missing or broken module instead of at container start.
# Not `import main`: it calls load_or_exit() and config.yaml is mounted, not baked.
RUN .venv/bin/python -c "import config, requestlog, performance_routing"

# config.yaml must be mounted at /app/config.yaml (not baked into image).
# Dokploy: use a File Mount (see README). Docker: -v ./config.yaml:/app/config.yaml

USER app

EXPOSE 4000

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:4000/health')"

CMD [".venv/bin/python", "main.py"]
