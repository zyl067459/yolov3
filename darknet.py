from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from util import count_parameters as count
from util import convert2cpu as cpu
from util import predict_transform

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


def parse_cfg(cfgfile):
    """
    获取配置文件



    返回块列表。每个块描述了神经系统中的一个块

    网络建设。块在列表中表示为字典
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     #store the lines in a list 将行存储在列表中
    lines = [x for x in lines if len(x) > 0] #get read of the empty lines  获取空行
    lines = [x for x in lines if x[0] != '#']  #去掉空行与以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]#去掉左右两边的空格

    # cfg文件中的每个块用[]括起来最后组成一个列表，一个block存储一个块的内容，即每个层用一个字典block存储。
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               #This marks the start of a new block 这是cfg文件中一个层(块)的开始
            if len(block) != 0: # 如果块内已经存了信息, 说明是上一个块的信息还没有保存
                blocks.append(block)# 那么这个块（字典）加入到blocks列表中去
                block = {} # 覆盖掉已存储的block,新建一个空白块存储描述下一个块的信息(block是字典)
            block["type"] = line[1:-1].rstrip()# 把cfg的[]中的块名作为键type的值
        else:
            key,value = line.split("=") #按等号分割
            block[key.rstrip()] = value.lstrip()#左边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对
    blocks.append(block)# 退出循环，将最后一个未加入的block加进去

    return blocks
#    print('\n\n'.join([repr(x) for x in blocks]))
# 配置文件定义了6种不同type
# 'net': 相当于超参数,网络全局配置的相关参数
# {'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}

# cfg = parse_cfg("cfg/yolov3.cfg")
# print(cfg)


import pickle as pkl

class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
    
    def forward(self, x):
        padded_x = F.pad(x, (0,self.pad,0,self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x
    

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
    
    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)
        return prediction
        

        


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
        
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H*stride, W*stride)
        return x
#       
        
class ReOrgLayer(nn.Module):
    def __init__(self, stride = 2):
        super(ReOrgLayer, self).__init__()
        self.stride= stride
        
    def forward(self,x):
        assert(x.data.dim() == 4)
        B,C,H,W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert(H % hs == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(H)
        assert(W % ws == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(W)
        x = x.view(B,C, H // hs, hs, W // ws, ws).transpose(-2,-3).contiguous()
        x = x.view(B,C, H // hs * W // ws, hs, ws)
        x = x.view(B,C, H // hs * W // ws, hs*ws).transpose(-1,-2).contiguous()
        x = x.view(B, C, ws*hs, H // ws, W // ws).transpose(1,2).contiguous()
        x = x.view(B, C*ws*hs, H // ws, W // ws)
        return x


def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing 捕获有关输入和预处理的信息
    # blocks[0]存储了cfg中[net]的信息，它是一个字典，获取网络输入和预处理相关信息
    module_list = nn.ModuleList() # module_list用于存储每个block,每个block对应cfg文件中一个块，类似[convolutional]里面就对应一个卷积块
    
    index = 0    #indexing blocks helps with implementing route  layers (skip connections) 索引块有助于实现路由层（跳过连接）

    
    prev_filters = 3 #初始值对应于输入数据3通道，用来存储我们需要持续追踪被应用卷积层的卷积核数量（上一层的卷积核数量（或特征图深度））
    
    output_filters = [] #我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。
    
    for x in blocks:
        module = nn.Sequential()# 这里每个块用nn.sequential()创建为了一个module,一个module有多个层
        
        if (x["type"] == "net"):
            continue
        
        #If it's a convolutional layer # 获取激活函数/批归一化/卷积层参数（通过字典的键获取值）
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])#卷积层后接BN就不需要bias
                bias = False
            except:
                batch_normalize = 0#卷积层后无BN层就需要bias
                bias = True
                
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
                
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True) # 给定参数负轴系数0.1
                module.add_module("leaky_{0}".format(index), activn)
            
            
            
        #If it's an upsampling layer
        #We use Bilinear2dUpsampling2. upsampling layer
           # 没有使用 Bilinear2dUpsampling
           # 实际使用的为最近邻插值

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
#            upsample = Upsample(stride)
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
        # route层的作用：当layer取值为正时，输出这个正数对应的层的特征，如果layer取值为负数，输出route层向后退layer层对应层的特征
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            
            #Start  of a route
            start = int(x["layers"][0])
            
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
                
            
            
            #Positive anotation
            if start > 0: 
                start = start - index
              # 若end>0，由于end= end - index，再执行index + end输出的还是第end层的特征
            if end > 0:
                end = end - index

            
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            # 若end<0，则end还是end，输出index+end(而end<0)故index向后退end层的特征。
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
                # 如果没有第二个参数，end=0，则对应下面的公式，此时若start>0，由于start = start - index，再执行index + start输出的还是第start层的特征;若start<0，则start还是start，输出index+start(而start<0)故index向后退start层的特征。

            else:
                filters= output_filters[index + start]
                        
            
        
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()#使用空的层，因为它还要执行一个非常简单的操作（加）。没必要更新 filters 变量,因为它只是将前一层的特征图添加到后面的层上而已。
            module.add_module("shortcut_{}".format(index), shortcut)
            
            
        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            
            module.add_module("maxpool_{}".format(index), maxpool)
        
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)# 锚点,检测,位置回归,分类，这个类见predict_transform中
            module.add_module("Detection_{}".format(index), detection)
        
            
            
        else:
            print("Something I dunno")
            assert False


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
        
    
    return (net_info, module_list)



class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)#调用parse_cfg函数
        self.net_info, self.module_list = create_modules(self.blocks)#调用create_modules函数
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

        
        
    def get_blocks(self):
        return self.blocks
    
    def get_module_list(self):
        return self.module_list

                
    def forward(self, x, CUDA):
        detections = []
        modules = self.blocks[1:]# 除了net块之外的所有，forward这里用的是blocks列表中的各个block块字典
        outputs = {}   #We cache the outputs for the route layer
        
        
        write = 0#write表示我们是否遇到第一个检测。write=0，则收集器尚未初始化，write=1，则收集器已经初始化，我们只需要将检测图与收集器级联起来即可。
        for i in range(len(modules)):        
            
            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                
                x = self.module_list[i](x)
                outputs[i] = x

                
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]
                
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                # 如果只有一层时。从前面的if (layers[0]) > 0:语句中可知，如果layer[0]>0，则输出的就是当前layer[0]这一层的特征,如果layer[0]<0，输出就是从route层(第i层)向后退layer[0]层那一层得到的特征
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                # 第二个元素同理
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                        
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    
                    
                    x = torch.cat((map1, map2), 1)#第二个参数设为 1,这是因为我们希望将特征图沿anchor数量的维度级联起来。
                outputs[i] = x
            
            elif  module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i+from_]# 求和运算，它只是将前一层的特征图添加到后面的层上而已
                outputs[i] = x
                
            
            
            elif module_type == 'yolo':        
                
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions#从net_info(实际就是blocks[0]，即[net])中get the input dimensions
                inp_dim = int (self.net_info["height"])
                
                #Get the number of classes
                num_classes = int (modules[i]["classes"])
                
                #Output the result
                x = x.data# 这里得到的是预测的yolo层feature map
                # 在util.py中的predict_transform()函数利用x(是传入yolo层的feature map)，得到每个格子所对应的anchor最终得到的目标
                # 坐标与宽高，以及出现目标的得分与每种类别的得分。经过predict_transform变换后的x的维度是(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                
                if type(x) == int:
                    continue

                
                if not write:  #if no collector has been intialised. 因为一个空的tensor无法与一个有数据的tensor进行concatenate操作，

                    detections = x #所以detections的初始化在有预测值出来时才进行，
                    write = 1   #用write = 1标记，当后面的分数出来后，直接concatenate操作即可。
                
                else:
                    '''
                                        变换后x的维度是(batch_size, grid_size * grid_size * num_anchors, 5 + 类别数量)，这里是在维度1上进行concatenate，即按照
                                        anchor数量的维度进行连接，对应教程part3中的Bounding
                                        Box
                                        attributes图的行进行连接。yolov3中有3个yolo层，所以
                                        对于每个yolo层的输出先用predict_transform()
                                        变成每行为一个anchor对应的预测值的形式(不看batch_size这个维度，x剩下的
                                        维度可以看成一个二维tensor)，这样3个yolo层的预测值按照每个方框对应的行的维度进行连接。得到了这张图处所有anchor的预测值，后面的NMS等操作可以一次完成
                                        '''
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i-1]



        try:
            return detections
        except:
            return 0


    def load_weights(self, weightfile):

        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype = np.int32, count = 5)# 这里读取first 5 values权重
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)#加载 np.ndarray 中的剩余权重，权重是以float32类型存储的

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"] # blocks中的第一个元素是网络参数和图像的描述，所以从blocks[1]开始读入

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model 将从weights文件中得到的权重bn_biases复制到model中(bn.bias.data
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:#如果 batch_normalize 的检查结果不是 True，只需要加载卷积层的偏置项
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)


                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def save_weights(self, savedfile, cutoff = 0):

        if cutoff <= 0:
            cutoff = len(self.blocks) - 1

        fp = open(savedfile, 'wb')

        # Attach the header at the top of the file
        self.header[3] = self.seen
        header = self.header

        header = header.numpy()
        header.tofile(fp)

        # Now, let us save the weights
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            if (module_type) == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    #If the parameters are on GPU, convert them back to CPU
                    #We don't convert the parameter to GPU
                    #Instead. we copy the parameter and then convert it to CPU
                    #This is done as weight are need to be saved during training
                    cpu(bn.bias.data).numpy().tofile(fp)
                    cpu(bn.weight.data).numpy().tofile(fp)
                    cpu(bn.running_mean).numpy().tofile(fp)
                    cpu(bn.running_var).numpy().tofile(fp)


                else:
                    cpu(conv.bias.data).numpy().tofile(fp)


                #Let us save the weights for the Convolutional layers
                cpu(conv.weight.data).numpy().tofile(fp)





#
#dn = Darknet('cfg/yolov3.cfg')
#dn.load_weights("yolov3.weights")
#inp = get_test_input()
#a, interms = dn(inp)
#dn.eval()
#a_i, interms_i = dn(inp)
