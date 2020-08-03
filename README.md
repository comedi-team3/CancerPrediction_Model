# Cancer Type Prediction Model
2020 Data Science Online Internship

## Installation
```
git clone git@github.com:comedi-team3/CancerPrediction_Model.git
cd CancerPrediction_Model
```

## Run

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
