import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
import cv2
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import time

def _read_image(image_file):
        print(image_file)
        image = cv2.imread(str(image_file))
        print(image)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
fileName = "/home/rdejana/jetson-inference/python/training/detection/ssd/data/fruit/test/0eb2a83700bd54d8.jpg"
f2 = "/home/rdejana/jetson-inference/python/training/detection/ssd/data/fruit/test/7cdf15687add8d3c.jpg"

img = _read_image(fileName)
i2 = _read_image(f2)

class_names = [name.strip() for name in open(args.label_file).readlines()]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
net = create_mobilenetv1_ssd(len(class_names), is_test=True)
print("Load Model")
net.load(args.trained_model)
net = net.to(DEVICE)
predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)

for x in range(6):
       start = time.time()
       boxes, labels, probs = predictor.predict(img)
       end = time.time()
       print("Time consumed in working: ",end - start)
       boxes, labels, probs = predictor.predict(i2)
       end = time.time()
       print("Time consumed in working: ",end - start)
