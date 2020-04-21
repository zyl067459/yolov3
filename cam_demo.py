from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl
import time
import multiprocessing as mp
import winsound

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(classes, colors, x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    if (cls == 1 or cls == 0):
        label = "{0}".format(classes[cls])
        # color = random.choice(colors)
        cv2.rectangle(img, c1, c2, colors[cls], 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, colors[cls], -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
        return img
    return

def write1(x,img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    m = 0
    #发现有显示7 先暂时排除
    if(cls == 7):
        return
    if(cls == 1):
        m = 1
    return m

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    parser.add_argument("--weights_path", dest='weights_path', type=str, default="checkpoints/yolov3_ckpt_4.pth",
                        help="path to weights file")
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="config/yolov3-custom.cfg", type=str)
    return parser.parse_args()


def image_put(q, user, pwd, ip, channel=1):
    # 根据摄像头设置IP及rtsp端口
    url = 'rtsp://admin:zyl123456@192.168.31.15:554/11'
    start = 0
    frames = 0
    # 读取视频流
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS,30)

    while True:
        is_opened, frame = cap.read()
        q.put(frame) if is_opened else None
        q.get() if q.qsize() > 1 else None


def image_get(q, window_name):
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    timeF = 20
    k = 0
    n = 0  # 计数
    frames = 0
    i = 0
    start = 0
    start = time.time()

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    CUDA = torch.cuda.is_available()

    num_classes = 2
    bbox_attrs = 5 + num_classes

    model = Darknet(args.cfgfile)
    if args.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(args.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(args.weights_path))

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32
    if CUDA:
        model.cuda()
    model.eval()
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
        #            im_dim = im_dim.repeat(output.size(0), 1)
        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        classes = load_classes('data/classes.names')
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: write(classes, colors, x, orig_im), output))
        list1 = list(map(lambda x: write1(x, orig_im), output))
        cv2.imshow(window_name, orig_im)#显示视频
        cv2.waitKey(1)
        frames += 1
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

        n = n + 1
        i += 1

        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            for j in range(0, len(list1)):
                if list1[j] == 1:
                    k = k + 1
                if list1[j] == 0:
                    k = 0
            if k != 0:
                cv2.imwrite('camera/{}.jpg'.format(i), orig_im)  # 当识别到未带安全帽时存储为图像

                # winsound.Beep(600, 1000)  # 当识别到未带安全帽时，调用蜂鸣器

def run_single_camera():
    # user_name, user_pwd, camera_ip = "admin", "admin123456", "172.20.114.196"
    user_name, user_pwd, camera_ip = "admin", "admin123456", "[fe80::3aaf:29ff:fed3:d260]"

    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=2)
    processes = [mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)),
                 mp.Process(target=image_get, args=(queue, camera_ip))]

    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    run_single_camera()





