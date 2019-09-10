FROM tensorflow/tensorflow:2.0.0rc0-gpu-py3

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /s4

# Usage: 
# docker build -t s4:latest .
# docker run -it --runtime=nvidia --rm -v $(pwd):/s4 -v $CITYSCAPES_DATASET:/data s4:latest python3 training.py --batch_size 8 --resolution 256,512 --input /data --epoch 100 --debug_freq 50 --log_dir logs
