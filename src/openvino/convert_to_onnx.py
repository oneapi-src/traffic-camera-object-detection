# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Converting model to onnx format
"""
# pylint: disable=E0401, C0103, R0914, C0415, E1101, C0413, C0301, E0402

import sys
sys.path.insert(0, '../yolov5')
from models.experimental import attempt_download
from models.yolo import Detect, Model
import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable


def main():
    """ Main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--outpath',
                        type=str,
                        required=False,
                        default='./openvino/openvino_models/openvino_onnx',
                        help='absolute path to save converted model. By default it '
                             'will be saved in "./openvino/openvino_models/openvino_onnx" folder')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=False,
                        default="yolov5s.pt",
                        help='Model Weights in ".pt" format')
    parser.add_argument('-mname',
                        '--model_name',
                        type=str,
                        required=False,
                        default="TrafficOD",
                        help="Name of the model to be created in \".onnx\" format, default \"TrafficOD\"")

    # Command line Arguments
    FLAGS = parser.parse_args()
    out_path = FLAGS.outpath
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model_name = FLAGS.model_name

    # Load model
    device = 'cpu'  # Device
    weights = FLAGS.weights

    imgsz = 640
    input_sz = torch.randn((1, 3, imgsz, imgsz))  # imgsz = 640
    dummy_input = Variable(input_sz)

    # Load model intel
    w = weights[0] if isinstance(weights, list) else weights
    ckpt = torch.load(attempt_download(w), map_location='cpu')
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()

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

    torch.onnx.export(model, dummy_input,
                      f"{out_path}/{model_name}_Onnx_Model.onnx", opset_version=11)


if __name__ == "__main__":
    main()
