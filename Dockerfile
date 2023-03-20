FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime


WORKDIR /app

COPY ./requirements.txt /app/requirements.txt


COPY . /app

COPY mongo-init.js /docker-entrypoint-initdb.d/

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y



CMD ["streamlit", "run", "SL_APP.py", "--server.address=0.0.0.0", "--server.port=8501"]

