FROM kaggle/python:latest

COPY environments/requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN cd /opt &&\
    git clone https://github.com/filipradenovic/cnnimageretrieval-pytorch

ENV PYTHONPATH $PYTHONPATH:/opt/models/research:/opt/cnnimageretrieval-pytorch

WORKDIR /workspace