# Use official TensorFlow image
FROM tensorflow/tensorflow:2.11.0

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install required packages
RUN pip install fastapi uvicorn joblib pydantic

# Expose port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
