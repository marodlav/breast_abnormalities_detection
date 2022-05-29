FROM --platform=linux/amd64 python:3.9-slim-buster
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir -p image_uploads image_results
COPY . .
CMD [ "python3", "app.py"]