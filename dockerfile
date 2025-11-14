# Lightweight Python image
FROM python:3.11-slim

# Working directory inside container
WORKDIR /app

# Copy only Flask API file(s)
COPY app.py /app/

# Install only Flask and Requests (for calling external services)
RUN pip install --no-cache-dir flask requests

# Expose the port Flask runs on
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]

