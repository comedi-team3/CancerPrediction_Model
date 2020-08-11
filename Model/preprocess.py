import pickle
import numpy as np
import torch

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

normal = []
for data, label in zip(X_final, X_types):
    if label == 'Normal Samples':
        normal.append(data)

print(normal)
print(len(normal))

sums = [0]*7100

for each in normal:
    for i in range(7100):
        sums[i] += each[i]

avg = []
for each in sums:
    avg.append(each / 713)

out = open('average.txt', 'w')

for each in avg:
    out.write(str(each)+'\n')

out.close()

'''
a = []
for data, label in zip(X_final, X_types):
    if label == 'TCGA-CHOL':
        a.append(np.concatenate((data, [33])))

print(len(a))
print(len(a[0]))

train_size = int(len(a)*0.7)
val_size = int(len(a)*0.2)
test_size = len(a) - (train_size + val_size)
print(train_size, val_size, test_size)

train_set, val_set, test_set = torch.utils.data.random_split(a, [train_size, val_size, test_size])
train_set, val_set, test_set = list(train_set), list(val_set), list(test_set)

# with open('./final_data/train.txt', 'w') as train:
#     np.savetxt(train, train_set)

# with open('./final_data/valid.txt', 'w') as valid:
#     np.savetxt(valid, val_set)

# with open('./final_data/test.txt', 'w') as test:
#     np.savetxt(test, test_set)

old_train = np.loadtxt('./final_data/train.txt')
old_valid = np.loadtxt('./final_data/valid.txt')
old_test = np.loadtxt('./final_data/test.txt')

with open('./final_data/train.txt', 'w') as train:
    data = np.concatenate((old_train, train_set))
    np.savetxt(train, data)

with open('./final_data/valid.txt', 'w') as valid:
    data = np.concatenate((old_valid, val_set))
    np.savetxt(valid, data)

with open('./final_data/test.txt', 'w') as test:
    data = np.concatenate((old_test, test_set))
    np.savetxt(test, data)
'''