FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY models/ ./models/

ARG TARGETARCH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update && apt-get --no-install-recommends install -y python3-opencv gcc

RUN --mount=type=cache,target=/root/.cache \
if [ "$TARGETARCH" = "arm64" ]; then \
	apt install -y libcamera; \
	pip install .[pi]; \
	else \
	pip install .; \
fi

EXPOSE 8080
CMD ["python", "-m", "src"]
