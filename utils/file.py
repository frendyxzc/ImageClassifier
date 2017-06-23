# -*- coding: utf-8 -*-
# author: frendy
# site: http://frendy.vip/
# time: 23/06/2017


def labelList():
    list = []

    file = open('./data/labels.txt')
    for line in file:
        list.append(line.split(" ")[1])

    return list