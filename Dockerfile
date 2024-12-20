# Use Python slim as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy over the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory into the container
COPY . .

# Run the two Python files sequentially as part of the image build process
# Note: These will execute only at build time, not when the container starts.
RUN python3 get_urls.py && python3 extract_data.py && python3 create_database.py

EXPOSE 5000

# Set the default command to run the app.py when the container starts
CMD ["python3", "chatbot/app.py"]
