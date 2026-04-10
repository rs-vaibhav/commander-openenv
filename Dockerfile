FROM python:3.10-slim

# Set environment limits to pass evaluation
ENV API_PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system -e .

COPY . .

EXPOSE 7860

# Pass the 2 vCPU 8 GB RAM limitation
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]