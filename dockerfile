FROM tensorflow/tensorflow:https://colab.research.google.com/drive/1G_7Gn7HE5qi4Tw94yj6V_TiLB4D0Zaco#scrollTo=4wolsK0cTdh6

WORKDIR /MLOPS_studies

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "infer.py"]
