# Predicting-cancer-type
Classifying the tumors and non-tumor gene expressions into **34 different types**: *33 cancer types + 1 normal sample type*.

1. [Installation](#installation)
2. [How to Run](#how-to-run)
3. [References](#references)

## Installation
Using data from the paper by [**Mostavi, et al.**](https://drive.google.com/open?id=1-Ib9jRNlfe0kqkYRdoBp3Q5aj9Q7EN3U)
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

## References
- Mostavi, Milad, et al. [**Convolutional neural network models for cancer type prediction based on gene expression**](https://link.springer.com/content/pdf/10.1186/s12920-020-0677-2.pdf). BMC Medical Genomics 13 (2020): 1-13.
