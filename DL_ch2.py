import torch
torch.tensor([[1.,-1.],[1.,-1.]])

import torch
print(torch.tensor([[1,2],[3,4]]))
print('------------------------')
print(torch.tensor([[1,2],[3,4]], dtype=torch.float64))


temp = torch.tensor([[1,2],[3,4]])
print(temp.numpy())
print('------------------------')
#temp = torch.tensor([[1,2],[3,4]], device="cuda:0") #GPU가 없다면 오류가 발생하므로 주석 처리하였습니다.
temp = torch.tensor([[1,2],[3,4]], device="cpu:0")
print(temp.to("cpu").numpy())

temp = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7])
print(temp[0], temp[1], temp[-1])
print('------------------------')
print(temp[2:5], temp[4:-1])

v = torch.tensor([1, 2, 3])
w = torch.tensor([3, 4, 6])
print(w - v)

temp = torch.tensor([
    [1, 2], [3, 4]
])

print(temp.shape)
print('------------------------')
print(temp.view(4,1))
print('------------------------')
print(temp.view(-1))
print('------------------------')
print(temp.view(1, -1))
print('------------------------')
print(temp.view(-1, 1))


#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r'C:/Users/Hong/Desktop/랩실 스터디/car_evaluation.csv')

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
dataset.output.value_counts().plot(kind='pie', autopct='%0.05f%%', 
                                   colors=['lightblue', 'lightgreen', 'orange', 'pink'], explode=(0.05,0.05,0.05,0.05))
# %%

categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')

price = dataset['price'].cat.codes.values # category type → 텐서
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)
categorical_data[:10]

categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
categorical_data[:10]

outputs = pd.get_dummies(dataset.output)
outputs = outputs.values
outputs = torch.tensor(outputs).flatten()

print(categorical_data.shape)
print(outputs.shape)

categorical_column_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]
print(categorical_embedding_sizes)
print(categorical_column_sizes)
min(50, (4+1)//2)


