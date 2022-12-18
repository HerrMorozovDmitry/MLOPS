FROM continuumio/miniconda3

WORKDIR /app

RUN echo "Conda Environment is Creating"
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "mlops_hw2", "/bin/bash", "-c"]

RUN echo "Awesome build is launched"
EXPOSE 5001
COPY titanic/train.csv /app/
COPY titanic/test.csv /app/
COPY titanic/gender_submission.csv /app/
COPY models.py /app/
COPY main.py /app/

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlops_hw2", "python", "/app/main.py"]