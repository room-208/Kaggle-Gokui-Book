FROM kaggle/python:latest

# Protocol Buffers
RUN cd /opt &&\
  wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip &&\
  unzip protoc-3.3.0-linux-x86_64.zip
ENV PATH_TO_PROTOC /opt

# DELG
RUN cd /opt &&\
  git clone https://github.com/tensorflow/models.git &&\
  cd /opt/models/research/delf &&\
  ${PATH_TO_PROTOC?}/bin/protoc delf/protos/*.proto --python_out=. &&\
  pip install -e . &&\
  cd /opt/models/research/delf/delf/python/delg &&\
  mkdir parameters && cd parameters &&\
  wget https://storage.googleapis.com/delf/r101delg_gldv2clean_20200914.tar.gz &&\
  tar -xvzf r101delg_gldv2clean_20200914.tar.gz
RUN mkdir parameters &&\
  cd parameters &&\
  wget https://storage.googleapis.com/delf/r101delg_gldv2clean_20200914.tar.gz &&\
  tar -xvzf r101delg_gldv2clean_20200914.tar.gz

# cirtorch
RUN cd /opt &&\
    git clone https://github.com/filipradenovic/cnnimageretrieval-pytorch
ENV PYTHONPATH $PYTHONPATH:/opt/models/research:/opt/cnnimageretrieval-pytorch

COPY environments/requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /workspace