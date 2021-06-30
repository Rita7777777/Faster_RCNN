from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from lxml import etree
import numpy as np


class_name = {
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


class VOC2012DataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, voc_root, shape = [800,800], txt_name: str = "train.txt"):
        self.root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)

        with open(txt_path) as read_path:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read_path.readlines()]

        self.class_name = class_name
        self.shape = shape


    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):

        #-----------------------   读  取  数  据   --------------------------#
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        # print(data["filename"])

        
        boxes = []
        labels = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            
            boxes.append(np.array([xmin, ymin, xmax, ymax]))
            labels.append(self.class_name[obj["name"]])
        
        boxes = np.array(boxes)
        label_data = np.array(labels)

        #-----------------------   数  据  增  强   --------------------------#
        iw, ih = image.size
        h,w = self.shape
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)
        img_data = np.transpose(image_data / 255.0, [2,0,1])


        #----- 对应改变box -----#
        if len(boxes) <= 0:
            ValueError("此图片没有检测框。")

        np.random.shuffle(boxes)
        boxes[:, [0,2]] = boxes[:, [0,2]]*nw/iw + dx
        boxes[:, [1,3]] = boxes[:, [1,3]]*nh/ih + dy
        boxes[:, 0:2][boxes[:, 0:2]<0] = 0
        boxes[:, 2][boxes[:, 2]>w] = w
        boxes[:, 3][boxes[:, 3]>h] = h
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w>1, boxes_h>1)]
        box_data = np.zeros((len(boxes),4))
        box_data[:len(boxes)] = boxes

        
        return img_data, box_data, label_data

    def get_height_and_width(self, idx):

        xml_path = self.xml_list[idx]
        with open(xml_path) as xml_file:
            xml_str = xml_file.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])

        return data_height, data_width

    def parse_xml_to_dict(self, xml):

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)
    return images, bboxes, labels

if __name__ == "__main__":
    root = '..'
    x = VOC2012DataSet(root)
    img, box, label = x[0]
    print(img.shape)
    print(box)
    print(label)