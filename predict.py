import colorsys
import copy
import os
import time
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from Faster_rcnn_frame import FasterRCNN
from utils import DecodeBox, get_new_img_size


class_n = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
    }

class_name = {v: k for k, v in class_n.items()}

parser = argparse.ArgumentParser(description=__doc__)

# 检测目标类别数(不包含背景)
parser.add_argument('--num_classes', default=20, type=int, help='num_classes')
# 文件保存地址
parser.add_argument('--train_model', default='./model_weight/train_model_weight.pth', help='path where to save')
# 是否用GPU
parser.add_argument('--cuda', default=True, type=bool, help='use gpu')
# iou
parser.add_argument('--iou', default=0.3, type=float, help='iou')
# confidence
parser.add_argument('--confidence', default=0.5, type=float, help='confidence')

args = parser.parse_args()



#-------------------------------------#
#             加载训练模型
#-------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FasterRCNN(args.num_classes)
# model_weights_path = './model_weight/voc_weights_vgg.pth'
model_weights_path = args.train_model
state_dict = torch.load(model_weights_path, map_location=device)
model.load_state_dict(state_dict)
model.cuda()
model.eval()


def detect_image(image):
    
    mean = torch.Tensor([0, 0, 0, 0]).repeat(args.num_classes+1)[None]
    std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(args.num_classes+1)[None]

    if args.cuda:
        mean = mean.cuda()
        std = std.cuda()
    decodebox = DecodeBox(std, mean, args.num_classes)

    image_shape = np.array(np.shape(image)[0:2])
    old_width, old_height = image_shape[1], image_shape[0]
    old_image = copy.deepcopy(image)
    
    #---------------------------------------------------------#
    #   给原图像进行resize，resize到短边为600的大小上
    #---------------------------------------------------------#
    width,height = get_new_img_size(old_width, old_height)
    image = image.resize([width,height], Image.BICUBIC)

    #-----------------------------------------------------------#
    #   图片预处理，归一化。
    #-----------------------------------------------------------#
    photo = np.transpose(np.array(image,dtype = np.float32)/255, (2, 0, 1))

    with torch.no_grad():
        images = torch.from_numpy(np.asarray([photo]))
        if args.cuda:
            images = images.cuda()

        roi_cls_locs, roi_scores, rois, _ = model(images)
    
        #-------------------------------------------------------------#
        #   利用classifier的预测结果对建议框进行解码，获得预测框
        #-------------------------------------------------------------#
        outputs = decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height = height, width = width, nms_iou = args.iou, score_thresh = args.confidence)
        #---------------------------------------------------------#
        #   如果没有检测出物体，返回原图
        #---------------------------------------------------------#
        if len(outputs)==0:
            raise ValueError('do not detect anything')
        outputs = np.array(outputs)
        bbox = outputs[:,:4]
        label = outputs[:, 4]
        conf = outputs[:, 5]

        bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
        bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height

    return bbox, label, conf


def draw_pred_img(image, bbox, label, conf):

    font = ImageFont.truetype(font='simhei.ttf',size=20) # size是字体的大小
    # 画框设置不同的颜色
    hsv_tuples = [(x / args.num_classes, 1., 1.)
                    for x in range(len(class_name))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    draw = ImageDraw.Draw(image)

    for i, c in enumerate(label):
        predicted_class = class_name[int(c)]
        score = conf[i]

        left, top, right, bottom = bbox[i]
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

        # 屏幕输出检测结果
        label = '{} {:.2f}'.format(predicted_class, score)
        
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, top, left, bottom, right)

        # 画检测物体框
        for i in range(4):   # 4是检测框框边的厚度
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[int(c)])

        # 画标签框
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[int(c)])
        # 在标签框里标注检测结果
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
   
    return image



img_path = './street.jpg'#input('Input image filename:')
image = Image.open(img_path)
image = image.convert("RGB")

bbox, label, conf = detect_image(image)
r_image = draw_pred_img(image, bbox, label, conf)
r_image.save('exam.jpg')
