FROM pytorch/pytorch:latest

WORKDIR /app

ARG TARGETARCH

COPY pyproject.toml ./
COPY src/ ./src/
COPY models/best_float32.tflite ./models/

# pip install dependencies for the detected architecture, with cache
RUN --mount=type=cache,target=/root/.cache \
	if [ "$TARGETARCH" = "arm64" ]; then \
	apt install -y libcamera; \
	pip install .[pi] --break-system-packages; \
	else \
	pip install . --break-system-packages; \
	fi

EXPOSE 8080
CMD ["python", "-m", "src"]
