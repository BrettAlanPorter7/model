FROM python:3.11-slim AS dependencies

WORKDIR /app

ARG TARGETARCH

# apt-update and install dependencies, with cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update && apt-get --no-install-recommends install -y python3-opencv gcc

FROM dependencies

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY models/ ./models/

# pip install dependencies for the detected architecture, with cache
RUN --mount=type=cache,target=/root/.cache \
if [ "$TARGETARCH" = "arm64" ]; then \
	apt install -y libcamera; \
	pip install .[pi]; \
	else \
	pip install .; \
fi

EXPOSE 8080
CMD ["python", "-m", "src"]
