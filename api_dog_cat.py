#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: api_dog_cat.py
Desc: 猫狗大战 ml模型 API 封装
Demo: 
    cd /home/work/piglab/webservice/service/ && nohup python api_dog_cat.py > log/api_dog_cat.log &
    #猫狗大战
    http://www.yanjingang.com:8021/piglab/image/dog_cat?img_file=/home/work/piglab/machinelearning/image/dog_cat/data/kaggle_infer/1.jpg

    ps aux | grep api_dog_cat.py |grep -v grep| cut -c 9-15 | xargs kill -9
Author: yanjingang(yanjingang@mail.com)
Date: 2018/12/28 23:08
"""

import sys
import os
import json
import logging
import tornado.ioloop
import tornado.web
import tornado.httpserver

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../')
sys.path.append(BASE_PATH)
#print(CUR_PATH, BASE_PATH)
from machinelearning.lib import utils
import infer as dogcat_infer

class ApiImageDogCat(tornado.web.RequestHandler):
    """API逻辑封装"""

    def get(self):
        """get请求处理"""
        try:
            result = self.execute()
        except:
            logging.error('execute fail ' + utils.get_trace())
            result = {'code': 1, 'msg': '查询失败'}
        logging.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                      + str(result['code']) + '][' + str(result['msg']) + '][' + str(result['data']) + ']')
        self.write(json.dumps(result))

    def post(self):
        """post请求处理"""
        try:
            result = self.execute()
        except:
            logging.error('execute fail ' + utils.get_trace())
            result = {'code': 1, 'msg': '查询失败'}
        logging.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                      + str(result['code']) + '][' + str(result['msg']) + ']')
        self.write(json.dumps(result))

    def execute(self):
        """执行业务逻辑"""
        logging.info('API REQUEST INFO[' + self.request.path + '][' + self.request.method + ']['
                      + self.request.remote_ip + '][' + str(self.request.arguments) + ']')
        img_file = self.get_argument('img_file', '')
        if img_file == '':
            return {'code': 2, 'msg': 'img_file不能为空'}
        res = {}

        try:
            ret, msg, res = dogcat_infer.infer(img_file)
            if ret != 0:
                logging.error('execute fail [' + img_file + '] ' + msg)
                return {'code': 4, 'msg': '查询失败'}
        except:
            logging.error('execute fail [' + img_file + '] ' + utils.get_trace())
            return {'code': 5, 'msg': '查询失败'}

        # 组织返回格式
        return {'code': 0, 'msg': 'success', 'data': res}


if __name__ == '__main__':
    """服务入口"""
    port = 8021

    # log init
    log_file = ApiImageDogCat.__name__.lower()  # + '-' + str(os.getpid())
    utils.init_logging(log_file=log_file, log_path=CUR_PATH)
    print("log_file: {}".format(log_file))

    # 路由
    app = tornado.web.Application(
        handlers=[
            (r'/piglab/image/dog_cat', ApiImageDogCat)
            ]
    )

    # 启动服务
    http_server = tornado.httpserver.HTTPServer(app, xheaders=True)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()

