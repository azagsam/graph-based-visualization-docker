FROM python:3.8

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8050

CMD ["python", "./main.py"]