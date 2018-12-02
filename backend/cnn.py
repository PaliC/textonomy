# class MultiDataset(object):
#     def __init__(self, dataset, num_outputs=1, transforms=None):
#         self.dataset = dataset
#         self.num_outputs = num_outputs
#         self.transforms = transforms

#     def __getitem__(self, idx):
#         # here comes the logic to convert a 1d index into a 
#         # self.num_output indices, each of size len(self.dataset)
#         individual_idx = []
#         for i in range(self.num_outputs):
#             individual_idx.append(idx % len(self.dataset))
#             idx = idx // len(self.dataset)
        
#         result = []
#         for i in reversed(idx):
#             result.append(self.dataset[i])

#         if self.transforms is not None:
#             result = self.transforms(result)

#         return result    

#     def __len__(self):
#         return len(self.dataset) ** self.num_outputs

import torch
from torch.utils.data import TensorDataset, DataLoader
from process import process, make_vectors

xtrain, xtest, ytrain, ytest = process()

X,Y, index_to_word, word_to_index = make_vectors(xtrain, ytrain)

t = torch.from_numpy(X)

dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
print(t)


import torch.nn as nn
import torch.nn.functional as f

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(1, 6, 5)
#         self.pool = nn.MaxPool1d(5)
#         self.conv2 = nn.Conv1d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()

# import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')
