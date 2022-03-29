FROM python:3.10-slim

COPY . .

RUN pip install '.[frontend]'

CMD  run_frontend