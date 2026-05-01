# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
# Install CPU‑only torch and torchvision first to avoid pulling CUDA packages
RUN pip install --no-cache-dir torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt --no-deps

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Default command runs Streamlit UI
CMD ["streamlit", "run", "streamlit_app.py"]