###
###
FROM python:3.9-slim-buster
ARG VERSION

# Install dependancies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app




COPY boot.sh /
RUN chmod +x /boot.sh

ADD Requirements.txt /app/Requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r Requirements.txt
RUN apt-get update && apt-get install -y ffmpeg

ADD . /app
RUN echo "${VERSION}" > version

EXPOSE 80
ENV WORKERS 3
ENV THREADS 3
ENV TIMEOUT 3000

CMD ["/boot.sh"]