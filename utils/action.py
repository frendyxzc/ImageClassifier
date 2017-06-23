# -*- coding: utf-8 -*-
# author: frendy
# site: http://frendy.vip/
# time: 23/06/2017

import numpy as np
import torch
import torch.nn
import torch.cuda
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from config import USE_CUDA

img_to_tensor = transforms.ToTensor()

# 分类
def inference(model, imgpath):
    model.eval()

    img = Image.open(imgpath)
    img = img.resize((224, 224))
    tensor = img_to_tensor(img)

    tensor = tensor.resize_(1, 3, 224, 224)
    if USE_CUDA:
        tensor = tensor.cuda()

    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()
    max_index = np.argmax(result_npy[0])

    return max_index


# 特征提取
def extract_feature(model, imgpath):
    model.fc = torch.nn.LeakyReLU(0.1)
    model.eval()

    img = Image.open(imgpath)
    img = img.resize((224, 224))
    tensor = img_to_tensor(img)

    tensor = tensor.resize_(1, 3, 224, 224)
    if USE_CUDA:
        tensor = tensor.cuda()

    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()

    return result_npy[0]