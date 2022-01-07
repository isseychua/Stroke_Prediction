# Define base image
FROM continuumio/miniconda3

COPY . .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "SPenv", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "SPenv", "python", "main_1.0.py"]