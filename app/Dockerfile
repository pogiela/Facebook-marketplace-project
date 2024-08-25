FROM python:3.12.4

WORKDIR /app

RUN apt-get update
RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

# Update setuptools and wheel
RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python3", "api.py"]
