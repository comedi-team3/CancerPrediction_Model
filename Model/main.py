import json
import time
import pickle
import torch
import numpy as np
import torch.nn as nn
from model import OneDimCNN
from collections import Counter
from matrix import batch_list, acc, epoch_time

def infer(num):
    model = OneDimCNN()
    model.load_state_dict(torch.load('./output_dir/1DCNN_0803_no_lr_2.pt'))

    model.eval()
    
    with open(f'./test_inputs/test_{num}_x.txt') as f:
        x = np.loadtxt(f)

    with open(f'./test_inputs/test_{num}_y.txt') as f:
        y = np.loadtxt(f)

    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.reshape(x.shape[0], 1, 100, 71)
    x = x.float()

    pred = model(x)

    y_hat = torch.max(pred, 1)[1]
    y = np.argmax(y, -1)

    print(pred, y_hat, y)

    return

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

    # for i in range(0, 100, 2):
    #     infer(i)

    # exit()
    
    second = open('./data/TCGA_new_pre_second.pckl', 'rb')
    [dropped_genes, dropped_genes_name, dropped_ens_id, sample_id, cancer_type, cancer_label] = pickle.load(second)
    second.close()

    first = open('./data/TCGA_new_pre_first.pckl', 'rb')
    [_, _, _, _, remaining_cancer_ids, remaining_normal_ids] = pickle.load(first)
    first.close()

    np.random.seed(1234)

    X_cancer = dropped_genes.iloc[:, remaining_cancer_ids].T.values
    X_normal = dropped_genes.iloc[:, remaining_normal_ids].T.values

    cancer_names = cancer_label[remaining_cancer_ids]
    normal_names = ['Normal Samples'] * len(X_normal)

    X_cancer = np.concatenate((X_cancer, X_normal))
    X_types = np.concatenate((cancer_names, normal_names))

    X_final = np.concatenate((X_cancer, np.zeros((len(X_cancer), 9))), axis=1)

    X_final = np.reshape(X_final, (-1, 71, 100))

    ## one hot encoding
    # classes = list(set(X_types))
    # X_types = [classes.index(each) for each in X_types]
    # X_types = np.eye(34)[X_types]

    train_size = int(0.6 * len(X_types)) #6631
    val_size = int((len(X_types) - train_size)/2) #2211
    test_size = len(X_types) - (train_size + val_size) # 2211

    train_set, val_set, test_set = torch.utils.data.random_split(list(zip(X_final, X_types)), [train_size, val_size, test_size])

    x_train, y_train = list(zip(*train_set))[0], list(zip(*train_set))[1]
    x_val, y_val = list(zip(*val_set))[0], list(zip(*val_set))[1]
    x_test, y_test = list(zip(*test_set))[0], list(zip(*test_set))[1]

   
    '''
    classes_json = json.dumps({'classes': classes})
    with open('classes_0803_2.json', 'w', encoding='utf-8') as f:
        json.dump(classes_json, f)
    
    for i in range(0, 100, 2):
        with open(f'./test_inputs/test_{i}_x.txt', 'w', encoding='utf-8') as f:
            np.savetxt(f, x_test[i])
        with open(f'./test_inputs/test_{i}_y.txt', 'w', encoding='utf-8') as f:
            np.savetxt(f, y_test[i])

    exit()
    '''

    batch_size = 128
    epoch = 100
    patience = 5

    model = OneDimCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    early_stop_check = 0
    best_valid_loss = float('inf')
    sorted_path = f'./output_dir/1DCNN_0806_no_lr_3.pt'

    for epoch in range(epoch):
        start_time = time.time()

        # train, validation
        train_loss, train_acc = train(x_train, y_train, batch_size, model)

        valid_loss, valid_acc = valid(x_val, y_val, batch_size, model)

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
