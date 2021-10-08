FROM tensorflow/tensorflow

WORKDIR /app
ADD . /app
RUN pip install tensorflow_datasets
CMD cd /app && python main.py