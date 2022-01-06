FROM python:3

WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt
CMD ["python","main_1.0.py"]