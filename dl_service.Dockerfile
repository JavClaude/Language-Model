FROM python:3.10-slim

COPY . .

RUN pip install '.[deep_learning_service]'

CMD  run_deep_learning_service