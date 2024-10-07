# https://docs.docker.com/go/dockerfile-reference/

# How to use Dockerfile

# First, using the terminal, move to where the Dockerfile is
# Then, build a local image based on the Dockerfile instructions
# docker build -t <image-name> .

# Then, it can be run as a container using
# docker run -it <image-name>

# Use the PyTorch base image
FROM pytorch/pytorch:latest

# Set the working directory inside the container
ARG APP_DIR=/app
WORKDIR ${APP_DIR}

# Copy requirements file and install package dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Command to run the program, if needed (in this case we don't)
#CMD ["cad"]