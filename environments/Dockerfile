FROM kaggle/python:latest

RUN cd /opt &&\
    git clone https://github.com/filipradenovic/cnnimageretrieval-pytorch

ENV PYTHONPATH $PYTHONPATH:/opt/models/research:/opt/cnnimageretrieval-pytorch

COPY environments/requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /workspace