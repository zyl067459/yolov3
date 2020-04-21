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
import winsound

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
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
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    # #发现有显示7 先暂时排除
    # if(cls == 7):
    #     return ;
    if(cls == 1 or cls == 0):
        label = "{0}".format(classes[cls])
        # color = random.choice(colors)
        cv2.rectangle(img, c1, c2,colors[cls], 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,colors[cls], -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img
    return

def write1(x,img):
    cls = int(x[-1])
    m = 0
    if(cls == 1):
        m = 1
    return m
def arg_parse():
    """
    Parse arguements to the detect module

    """


    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    parser.add_argument("--weights_path", dest='weights_path', type=str, default="checkpoints/yolov3_ckpt_4.pth",
                        help="path to weights file")
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="config/yolov3-custom.cfg", type=str)
    return parser.parse_args()



if __name__ == '__main__':
    # cfgfile = "cfg/yolov3.cfg"
    # weightsfile = "yolov3.weights"
    num_classes = 2

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()



    i = 0
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

    videofile = 'video.avi'

    # 根据摄像头设置IP及rtsp端口
    url = 'rtsp://admin:zyl123456@192.168.31.15:554/11'

    # 读取视频流
    # cap = cv2.VideoCapture(url)
    # 电脑摄像头
    cap = cv2.VideoCapture(0)
    timeF = 20
    assert cap.isOpened(), 'Cannot capture source'
    k = 0
    frames = 0
    start = time.time()
    #这样才能在断开连接时重新连接上
    while True:
        ret, frame = cap.read()
        if ret:

            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1,2)


            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()


            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue



            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim

#            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            classes = load_classes('data/classes.names')
            colors = pkl.load(open("pallete", "rb"))
            # m = write1(x,orig_im)
            # x= write(x, orig_im)
            # print(x)
            list(map(lambda x: write(x, orig_im), output))
            list1 = list(map(lambda x: write1(x, orig_im), output))
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
            if (frames % timeF == 0):  # 每隔timeF帧进行存储操作
                i = i + 1
                for j in range(0,len(list1)):
                    if list1[j] == 1:
                        k = k + 1
                    if list1[j] == 0:
                        k = 0
                if k != 0:
                    cv2.imwrite('camera/{}.jpg'.format(i), orig_im)  # 当识别到未带安全帽时存储为图像

                    winsound.Beep(600, 1000)#当识别到未带安全帽时，调用蜂鸣器

        else:
            #断开连接时 重新连接
            # st = time.time()
            cap = cv2.VideoCapture(url)
            # print("tot time lost due to reinitialization : ", time.time() - st)
            continue





