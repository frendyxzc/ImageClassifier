# -*- coding: utf-8 -*-
# author: frendy
# site: http://frendy.vip/
# time: 23/06/2017

import sys
from models import resnet34, vgg16
from utils.action import inference, extract_feature

IMAGE = sys.argv[1]


if __name__ == "__main__":
    model, labels = resnet34()
    # model, labels = vgg16()
    img = IMAGE

    # print(extract_feature(model, img))
    print(labels[inference(model, img)])