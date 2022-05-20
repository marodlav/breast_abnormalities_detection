FROM python:3.10-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir -p image_uploads image_results
COPY . .
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=80"]