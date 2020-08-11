import torch
import numpy as np
from flask import jsonify
from model import OneDimCNN

def infer(data):
    model = OneDimCNN()
    model.load_state_dict(torch.load('./output_dir/1DCNN_0809_15e-4.pt'))

    model.eval()
    
    with open(f'./input/{data}') as f:
        x = np.loadtxt(f)

    with open(f'./input/{data}') as f:
        y = np.loadtxt(f)

    with open('labels.txt') as f:
        labels = f.readlines()

    with open('cancer_names.txt') as f:
        names = f.readlines()

    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.reshape(x.shape[0], 1, 100, 71)
    x = x.float()

    pred = model(x)

    y_hat = torch.max(pred, 1)[1]
    y = np.argmax(y, -1)

    return labels[y], names[y]

def top_ten(data):
    curr = np.loadtxt(f'./inputs/{data}')

    with open('average.txt') as f:
        avgs = f.readlines()

    with open('gene_names.txt') as f:
        genes = f.readlines()

    abs_diff = []
    for i in range(7100):
        abs_diff.append(abs(curr[i] - float(avgs[i])))
    
    top_10_idx = np.argsort(abs_diff)[-10:]
    top_10_labels = [genes[i].strip() for i in top_10_idx]
    top_10_normal = [float(avgs[i]) for i in top_10_idx]
    top_10_input = [curr[i] for i in top_10_idx]
    
    return top_10_labels, top_10_normal, top_10_input

def model_inference(data):
    cohort, name = infer(data)
    top_labels, top_normal, top_input = top_ten(data)

    return cohort, name, top_labels, top_normal, top_input

if __name__=="__main__":
    #infer(40)
    #top_ten('10_x.txt')
    model_inference('1023_x.txt')