# Use official Python image
FROM python:3.6.13

# Set the working directory
WORKDIR /

# Copy all files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the app
CMD ["python", "index.py"]

EXPOSE 10000
