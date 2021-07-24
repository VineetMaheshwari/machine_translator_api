FROM python:3.8

WORKDIR /api

COPY requirements.txt /api
RUN pip install -r requirements.txt

COPY . /api

CMD python api.py