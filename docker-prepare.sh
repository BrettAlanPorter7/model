docker build . -t drone-detection
docker run -t -p 8080:8080 --device /dev/video0:/dev/video0 drone-detection
