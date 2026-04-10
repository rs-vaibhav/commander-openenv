FROM python:3.10-slim

# Install uv for blazing fast package management
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy dependency file first for caching
COPY pyproject.toml .

# Install dependencies using uv
RUN uv pip install --system -e .

# Copy the rest of the application
COPY . .

# Expose the standard Hugging Face port
EXPOSE 7860

# Start the FastAPI server using uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]