#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: infer.py
Desc: 图像分类预测（resnet|vgg）
Author:yanjingang(yanjingang@mail.com)
Date: 2018/12/26 22:11
Cmd: python infer.py ./data/kaggle_infer/1.jpg
"""

from __future__ import print_function
import sys
import os
import getopt
import logging
import paddle.fluid as fluid
from paddle.fluid.contrib.trainer import *
from paddle.fluid.contrib.inferencer import *
import numpy

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../../')
sys.path.append(BASE_PATH)
#print(CUR_PATH, BASE_PATH)
from machinelearning.lib import utils
import train as dogcat_train

# label名称
label_list = ["cat", "dog"]


def infer(img_file='', params_dirname=CUR_PATH, use_cuda=False):
    """使用模型测试"""
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return 1, 'compiled is not with cuda', {}
    if img_file == '':
        return 1, 'file_name is empty', {}

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    logging.debug('param_path:' + params_dirname + '/model')
    inferencer = Inferencer(infer_func=dogcat_train.image_classification_network, param_path=params_dirname + '/model', place=place)

    # 预测
    img = utils.load_rgb_image(img_file)
    # print(img.shape)
    result = inferencer.infer({'img': img})
    result = numpy.where(result[0][0] > 0.05, result[0][0], 0)  # 概率<5%的直接设置为0
    print(result)
    label = int(numpy.argmax(result))
    label_name = label_list[label]
    weight = result[label]

    print("*img: %s" % img_file)
    print("*label: %d" % label)
    print("*label_name: %s" % label_name)
    print("*label weight: %f" % weight)

    return 0, '', {'img': img_file, 'label': label, 'label_name': label_name, 'weight': str(weight)}


def kaggle_infer(kaggle_path=CUR_PATH + '/data/kaggle_infer100/'):
    """
    kaggle测试集预测
    cat kaggle_infer.ori.csv|sort -n >kaggle_infer.csv
    """
    # kaggle label (0:cat, 1:dog)
    kaggle_labes = [
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
    ]
    errs = []
    print(len(kaggle_labes))
    # res file
    fo = open(kaggle_path[:-1] + '.ori.csv', "w")
    fo.write("id,label\n")
    # test infer
    imgs = os.listdir(kaggle_path)
    for i in xrange(len(imgs)):
        # print(imgs[i])
        id = int(imgs[i].split('.')[0])
        label = -1
        label_name = ''
        weight = 0.0
        ret, msg, res = infer(kaggle_path + imgs[i])
        # print(res)
        if ret == 0:
            label = res['label']
            label_name = res['label_name']
            weight = res['weight']

        # acc手工标注准确率自动检查
        if id <= len(kaggle_labes):
            if label != kaggle_labes[id - 1]:
                errs.append(id)

        line = str(id) + "," + str(label) + "," + str(label_name) + "," + str(weight) + "\n"
        print("*kaggle_infer: " + line)
        fo.write(line)
        # break

    # acc print
    succ = len(kaggle_labes) - len(errs)
    print("*kaggle_infer acc : " + str(round(succ / len(kaggle_labes),4)) + "\t errs:" + str(errs))

    fo.close()


if __name__ == '__main__':
    """infer test"""
    img_file = CUR_PATH + '/data/kaggle_infer/5.jpg'
    opts, args = getopt.getopt(sys.argv[1:], "p:", ["file_name="])
    if len(args) > 0 and len(args[0]) > 4:
        img_file = args[0]

    # infer
    ret = infer(img_file)
    print(ret)

    # kaggle test infer
    # kaggle_infer()
