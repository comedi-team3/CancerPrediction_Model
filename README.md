# Predicting-cancer-type
2020 Data Science Online Internship
1. [Installation](#installation)
2. [How to Run](#how-to-run)

## Installation
Using data from the paper by [**Mostavi et al.**](https://drive.google.com/open?id=1-Ib9jRNlfe0kqkYRdoBp3Q5aj9Q7EN3U).
```
git clone git@github.com:comedi-team3/CancerPrediction_Model.git
cd CancerPrediction_Model
```

## How to Run

### Running locally
```python
python main.py
```

### Running through Docker
```python
docker build -t cancer_prediction .
docker run -it --name cancerprediction_model -v ~/CancerPrediction_Model:/workspace cancer_prediction /bin/bash
cd Model
python main.py
```
