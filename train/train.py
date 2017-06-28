# -*- coding: utf-8 -*-
# author: frendy
# site: http://frendy.vip/
# time: 28/06/2017

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from config import MODEL_PATH, DATA_PATH
from models.model import transform, Net
from data import loadTrainData

trainset, trainloader = loadTrainData()

net = Net()
if os.path.exists(MODEL_PATH):
    net.load_state_dict(torch.load(MODEL_PATH))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# save model
torch.save(net.state_dict(), MODEL_PATH)