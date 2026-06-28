FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SDL_VIDEODRIVER=dummy
ENV PYGAME_HIDE_SUPPORT_PROMPT=1
ENV PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY web_play/requirements-web.txt /app/web_play/requirements-web.txt
RUN pip install --no-cache-dir -r /app/web_play/requirements-web.txt

COPY . /app

EXPOSE 7860

CMD ["sh", "-c", "python -m web_play.server --host 0.0.0.0 --port ${PORT:-7860}"]
