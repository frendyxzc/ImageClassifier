# -*- coding: utf-8 -*-
# author: frendy
# site: http://frendy.vip/
# time: 23/06/2017

import torchvision.models as models
from config import USE_CUDA
from utils.file import labelList


def resnet34():
    model = models.resnet34(pretrained=True)
    if USE_CUDA:
        model.cuda()

    labels = labelList()

    return model, labels

