import json
import time
import pickle
import torch
import numpy as np
import torch.nn as nn
from model import OneDimCNN
from collections import Counter
from matrix import batch_list, acc, epoch_time

def one_hot(data):
    n_values = np.max(data) + 1

    return np.eye(n_values)[data]

def data_split(data):
    x, y = [], []
    
    for each in data:
        x.append(each[:-1])
        y.append(int(each[-1]))
    
    y = one_hot(y)

    return x, y

def infer(num):
    model = OneDimCNN()
    model.load_state_dict(torch.load('./output_dir/1DCNN_0809_15e-4.pt'))

    model.eval()
    
    with open(f'./inputs/{num}_x.txt') as f:
        x = np.loadtxt(f)

    with open(f'./inputs/{num}_y.txt') as f:
        y = np.loadtxt(f)

    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.reshape(x.shape[0], 1, 100, 71)
    x = x.float()

    pred = model(x)

    y_hat = torch.max(pred, 1)[1]
    y = np.argmax(y, -1)

    print(y_hat, y)

    return y_hat == y

def train(x, y, batch_size, model):
    x_batches = batch_list(x, batch_size)
    y_batches = batch_list(y, batch_size)

    total_loss, iter_num, train_acc = 0, 0, 0
    model.train()

    for step, data in enumerate(zip(x_batches, y_batches)):
        inputs, labels = np.array(data[0]), torch.tensor(data[1])
        
        inputs = inputs.reshape(inputs.shape[0], 1, 100, 71)
        inputs = torch.from_numpy(inputs).float() #torch.Size([128, 1, 100, 71])

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, np.argmax(labels, -1))
        loss.backward()
        optimizer.step()

        total_loss += loss
        iter_num += 1

        with torch.no_grad():
            tr_acc = acc(torch.max(outputs, 1)[1], np.argmax(labels, -1))
        train_acc += tr_acc

    return total_loss / iter_num, train_acc / iter_num

def valid(x, y, batch_size, model):
    x_batches = batch_list(x, batch_size)
    y_batches = batch_list(y, batch_size)

    total_loss, iter_num, val_acc = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for step, data in enumerate(zip(x_batches, y_batches)):
            inputs, labels = np.array(data[0]), torch.tensor(data[1])

            inputs = inputs.reshape(inputs.shape[0], 1, 100, 71)
            inputs = torch.from_numpy(inputs).float() #torch.Size([128, 1, 100, 71])

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, np.argmax(labels, -1))

            total_loss += loss
            iter_num += 1

            v_acc = acc(torch.max(outputs, 1)[1], np.argmax(labels, -1))
            val_acc += v_acc

        return total_loss / iter_num, val_acc / iter_num

def test(x, y, batch_size, model):
    x_batches = batch_list(x, batch_size)
    y_batches = batch_list(y, batch_size)

    total_loss, iter_num, test_acc = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for step, data in enumerate(zip(x_batches, y_batches)):
            inputs, labels = np.array(data[0]), torch.tensor(data[1])

            inputs = inputs.reshape(inputs.shape[0], 1, 100, 71)
            inputs = torch.from_numpy(inputs).float() #torch.Size([128, 1, 100, 71])

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, np.argmax(labels, -1))

            total_loss += loss
            iter_num += 1

            te_acc = acc(torch.max(outputs, 1)[1], np.argmax(labels, -1))
            test_acc += te_acc

        return total_loss / iter_num, test_acc / iter_num

if __name__=="__main__":
    
    print('Loading data ...')
    train_data = np.loadtxt('./final_data/train.txt')
    valid_data = np.loadtxt('./final_data/valid.txt')
    test_data = np.loadtxt('./final_data/test.txt')

    np.random.seed(1234)
    
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    print('Loading done !')
    print('Splitting data ...')

    x_train, y_train = data_split(train_data)
    x_valid, y_valid = data_split(valid_data)
    x_test, y_test = data_split(test_data)

    print('Spliiting done !')
 
    batch_size = 128
    epoch = 200
    patience = 5

    model = OneDimCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=15e-4)

    early_stop_check = 0
    best_valid_loss = float('inf')
    sorted_path = f'./output_dir/1DCNN_0809_15e-4_2.pt'

    for epoch in range(epoch):
        start_time = time.time()

        # train, validation
        train_loss, train_acc = train(x_train, y_train, batch_size, model)

        valid_loss, valid_acc = valid(x_valid, y_valid, batch_size, model)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if early_stop_check == patience:
            print("\n*****Early stopping*****\n")
            break

        # saving model when current loss is lesser than previous loss
        if valid_loss < best_valid_loss:
            early_stop_check = 0
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), sorted_path)
            print(f'\n\t## SAVE valid_loss: {valid_loss:.3f} | valid_acc: {valid_acc:.3f} ##')
        else:
            early_stop_check += 1

        # print loss and acc
        print(f'\n\t==Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s==')
        print(f'\t==Train Loss: {train_loss:.3f} | Train acc: {train_acc:.3f}==')
        print(f'\t==Valid Loss: {valid_loss:.3f} | Valid acc: {valid_acc:.3f}==\n')

    test_loss, test_acc = test(x_test, y_test, batch_size, model)
    print(f'\n\t==Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}==\n')
