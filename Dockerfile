FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime


WORKDIR /app

COPY ./requirements.txt /app/requirements.txt


COPY . /app


RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y



ENTRYPOINT ["streamlit", "run", "SL_APP.py", "--server.port=8501", "--server.address=0.0.0.0"]
