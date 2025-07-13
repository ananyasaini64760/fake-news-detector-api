# Use TensorFlow base image
FROM tensorflow/tensorflow:2.11.0

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install fastapi uvicorn joblib pydantic

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
