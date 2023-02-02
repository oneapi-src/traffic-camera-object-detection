# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=E1101, E1120, C0301, E0401, I1101, C0103, C0411, R0913, R0914, R0915, C1801, C0200

"""
Inference FP32 and INT8
"""

import argparse
import os
import itertools
import time
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from utils.plots import Annotator, colors
from utils.dataloaders import create_dataloader
from utils.metrics import ap_per_class
from utils.general import (check_dataset, check_img_size, check_yaml, colorstr, non_max_suppression, scale_boxes,
                           xywh2xyxy)
from models.yolo import Detect, Model
from models.experimental import attempt_download
from val import process_batch


def do_overlap(l1, r1, l2, r2):
    """
    Returns true if two rectangles(l1, r1)
    and (l2, r2) overlap
    """
    # if rectangle has area 0, no overlap
    if l1[0] == r1[0] or l1[1] == r1[1] or r2[0] == l2[0] or l2[1] == r2[1]:
        return False
    # If one rectangle is on left side of other
    if l1[0] > r2[0] or l2[0] > r1[0]:
        return False
    # If one rectangle is above other
    if r1[1] > l2[1] or r2[1] > l1[1]:
        return False
    return True


def point_in_rect(point, rect):
    """
    check whether centre is inside prediction
    """
    (x1, y1, x2, y2) = rect
    (x, y) = point
    if x1 < x < x2:
        if y1 < y < y2:
            return True
    return False


def distancing(people_coords=None, vehicle_coords=None, img=None, dist_thres_lim=(200, 250)):
    """
    Analyze the distance between objects and predict possible accident scenarios
    """
    # Plot lines connecting people
    already_red = {}  # dictionary to store if a plotted rectangle has already been labelled as high risk
    centers_people = []
    for i_c in people_coords:
        centers_people.append(((int(i_c[2]) + int(i_c[0])) // 2, (int(i_c[3]) + int(i_c[1])) // 2))
    centers_vehicle = []
    for i_v in vehicle_coords:
        centers_vehicle.append(((int(i_v[2]) + int(i_v[0])) // 2, (int(i_v[3]) + int(i_v[1])) // 2))
    for j in centers_people + centers_vehicle:
        already_red[j] = 0

    x_combs = list(itertools.product(people_coords, vehicle_coords))
    radius = 10
    thickness = 5

    for x in x_combs:
        xyxy1, xyxy2 = x[0], x[1]
        cntr1 = ((int(xyxy1[2]) + int(xyxy1[0])) // 2, (int(xyxy1[3]) + int(xyxy1[1])) // 2)
        cntr2 = ((int(xyxy2[2]) + int(xyxy2[0])) // 2, (int(xyxy2[3]) + int(xyxy2[1])) // 2)
        dist = ((cntr2[0] - cntr1[0]) ** 2 + (cntr2[1] - cntr1[1]) ** 2) ** 0.5

        overlap1 = point_in_rect(cntr1, x[1])
        overlap2 = point_in_rect(cntr2, x[0])
        if overlap1 and overlap2:
            continue

        if dist_thres_lim[0] < dist < dist_thres_lim[1]:
            color = (0, 255, 255)
            risk_label = "Low Risk "
            cv2.line(img, cntr1, cntr2, color, thickness)
            if already_red[cntr1] == 0:
                cv2.circle(img, cntr1, radius, color, -1)
            if already_red[cntr2] == 0:
                cv2.circle(img, cntr2, radius, color, -1)
            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                cntr = ((int(xy[2]) + int(xy[0])) // 2, (int(xy[3]) + int(xy[1])) // 2)
                if already_red[cntr] == 0:
                    c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(risk_label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, risk_label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                                lineType=cv2.LINE_AA)
            # print("Low Risk Found")

        elif dist < dist_thres_lim[0]:
            color = (0, 0, 255)
            risk_label = "High Risk"
            already_red[cntr1] = 1
            already_red[cntr2] = 1
            cv2.line(img, cntr1, cntr2, color, thickness)
            cv2.circle(img, cntr1, radius, color, -1)
            cv2.circle(img, cntr2, radius, color, -1)
            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(risk_label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, risk_label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                            lineType=cv2.LINE_AA)
            # print("High Risk Found")
    return img


class Dataset:
    """Creating Dataset class for getting Image and labels"""

    def __init__(self, dataloader, batch_s):
        self.dataloader = dataloader
        self.batch_size = batch_s
        self.im = None
        for _, (img, target, _, _) in enumerate(self.dataloader):
            self.im = img.to(device)
            self.im = self.im.float()
            self.im /= 255  # 0 - 255 to 0.0 - 1.0
            self.targets = []
            for d_b in range(self.batch_size):
                new_s = []
                for d_s in np.array(target[:, 0]):
                    if int(d_s) == d_b:
                        new_s.append(d_s)
                self.targets.append(new_s)
            break

    def __getitem__(self, index):
        return self.im[index], self.targets[index]

    def __len__(self):
        return self.batch_size


def post_processing_yolo(predicts, im_orig, im_norm, image_pth, image_flag, count):
    """
    main post processing function
    """
    person_coords = []
    vehicle_cords = []
    # Process predictions
    for _, det in enumerate(predicts):  # per image
        # copying the original image
        im0 = im_orig.copy()
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im_norm.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for _ in det[:, -1].unique():
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    vehicles = ["car", "bus", "truck"]
                    if names[c] == "person":
                        person_coords.append(xyxy)
                        label = f'{names[c]} {conf:.2f}'
                        # print(label)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    elif names[c] in vehicles:
                        vehicle_cords.append(xyxy)
                        label = f'{names[c]} {conf:.2f}'
                        # print(label)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    else:
                        continue
        image_op = distancing(person_coords, vehicle_cords, im0,
                              dist_thres_lim=(int(im0.shape[0] * 0.3), int(im0.shape[0] * 0.35)))

        img_cls = "_vehicle"
        if image_flag and (person_coords or vehicle_cords):
            if person_coords and vehicle_cords:
                img_cls = "_people&vehicles"
            elif person_coords:
                img_cls = "_people"

            file_opath = image_pth + "/output"
            # Create directories for the paths provided if not existing
            os.makedirs(file_opath, exist_ok=True)
            # saving the image
            cv2.imwrite(file_opath + "/" + str(count) + img_cls + "_image.jpg", image_op)


# Define the command line arguments to input the Hyperparameters - batchsize & Learning Rate
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default='./deploy.yaml',
                        help='Yaml file for quantizing model, default is "./deploy.yaml"')
    parser.add_argument('-d',
                        '--data_yaml',
                        type=str,
                        required=False,
                        default='./data/VOC.yaml',
                        help='Absolute path to the data yaml file containing configurations')
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=1,
                        help='batchsize for the dataloader....default is 1')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=False,
                        default="yolov5s.pt",
                        help='Model Weights ".pt" format')
    parser.add_argument('-i',
                        '--intel',
                        type=int,
                        required=False,
                        default=0,
                        help='Run Intel optimization (Ipex) when 1....default is 0')
    parser.add_argument('-int8inc',
                        type=int,
                        required=False,
                        default=0,
                        help='Run INC quantization when 1....default is 0')
    parser.add_argument('-qw',
                        '--quant_weights',
                        type=str,
                        required=False,
                        default="./inc_compressed_model/output",
                        help='Quantization Model Weights folder containing ".pt" format model')
    parser.add_argument('-si',
                        '--save_image',
                        type=int,
                        required=False,
                        default=0,
                        help='Save images in the save image path specified if 1, default 0')
    parser.add_argument('-sip',
                        '--save_image_path',
                        type=str,
                        required=False,
                        default="./saved_images",
                        help='Path to save images after post processing/ detected results')

    # Command line Arguments
    FLAGS = parser.parse_args()
    batch_size = FLAGS.batchsize
    config_path = FLAGS.config
    data_yml = FLAGS.data_yaml
    intel = FLAGS.intel
    int8inc = FLAGS.int8inc
    quant_weights = FLAGS.quant_weights
    save_image = FLAGS.save_image
    save_img_path = FLAGS.save_image_path

    data = check_yaml(data_yml)  # check YAML
    device = 'cpu'  # Device
    weights = FLAGS.weights

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    ckpt = torch.load(attempt_download(w), map_location='cpu')
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()
    stride = 32
    names = dict(enumerate(ckpt.names))

    model = ckpt.fuse().eval()
    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = True
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

            # Configure
    imgsz = check_img_size(640, s=stride)
    dummy_inp = torch.zeros(1, 3, imgsz, imgsz).to(device, non_blocking=True)

    data = check_dataset(data)  # check
    single_cls = False
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    task = 'val'

    dataloader_l = create_dataloader(data[task],
                                     imgsz,
                                     batch_size,
                                     stride,
                                     single_cls=single_cls,
                                     pad=0.5,
                                     rect=False,  # pt
                                     workers=8,
                                     prefix=colorstr(f'{task}: '))[0]

    names = data['names']

    if intel:
        import intel_extension_for_pytorch as ipex

        print('IPEX optimization enabled')
        model = ipex.optimize(model, inplace=True)
    elif int8inc:
        import intel_extension_for_pytorch as ipex
        from neural_compressor.utils.pytorch import load

        model = load(quant_weights, model)
        model = model.eval()
        model = ipex.optimize(model, inplace=True)
        print('IPEX + INC optimization enabled')
    else:
        print('Stock model')

    model = torch.jit.trace(model, dummy_inp, check_trace=False, strict=False)
    model = torch.jit.freeze(model)

    # Timing Analysis
    AVG_TIME = 0
    ACCURACY = []
    MAX_NUM_IERATIONS = 1000
    WARMUP = 10
    COUNT = 0

    # Evaluation
    iouv = torch.linspace(0.5, 0.95, 10, device='cpu')
    niou = iouv.numel()
    save_dir = Path('')
    plots = False
    device = 'cpu'
    task = 'val'
    conf_thres = 0.001  # confidence threshold
    iou_thres = 0.6  # NMS IoU threshold
    max_det = 300
    seen = 0
    cnt = 0

    lb, stats = [], []
    mAp = 0.0

    with torch.no_grad():
        for _ in range(len(dataloader_l)):
            (im, targets, paths, shapes) = next(iter(dataloader_l))
            COUNT += 1

            if COUNT > WARMUP:
                if COUNT > MAX_NUM_IERATIONS + WARMUP:
                    break
                im_o = np.array(im).copy()
                im = im.to(device)
                im = im.float()
                im /= 255  # 0 - 255 to 0.0 - 1.0
                im = nnf.interpolate(im, size=(640, 640), mode='bicubic', align_corners=False)
                nb, _, height, width = im.shape
                targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
                start_time = time.time()
                preds = model(im)
                pred_time = time.time() - start_time
                conf_thres = 0.25
                iou_thres = 0.45
                classes = None
                preds = non_max_suppression(preds, conf_thres, iou_thres, classes, max_det=1000)
                # Metrics
                for si, pred in enumerate(preds):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                    _, shape = Path(paths[si]), shapes[si][0]
                    correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                    seen += 1

                    if npr == 0:
                        if nl:
                            stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                        continue
                    predn = pred.clone()
                    # Evaluate
                    if nl:
                        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                        correct = process_batch(predn, labelsn, iouv)
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

                for i in range(len(im_o)):
                    im_rgb = cv2.cvtColor(im_o[i].transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
                    if save_image:
                        cnt += 1
                    post_processing_yolo([preds[i]], im_rgb, im[i][None, :], save_img_path, save_image, cnt)
                AVG_TIME += pred_time

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

        if len(stats) and stats[0].any():
            _, _, _, _, _, ap, _ = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            ap = ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mAp = ap.mean()

    print("\nMean Average Precision for all images is ", mAp)
    print("Batch Size used here is ", batch_size)
    print("Average Inference Time Taken --> ", (AVG_TIME / COUNT), "for images ::", COUNT)
