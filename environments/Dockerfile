FROM kaggle/python:latest

COPY environments/requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /workspace