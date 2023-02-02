# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
INC QUANTIZATION model saving
"""
# pylint: disable=C0103,C0301,E0401,R0914,W0622,C1801,R0912,W1201,W1202,R0915,E0602

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from neural_compressor.experimental import Quantization, common
from utils.dataloaders import create_dataloader
from utils.metrics import ap_per_class
from utils.callbacks import Callbacks
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_yaml,
                           colorstr, non_max_suppression, scale_boxes, xywh2xyxy)
from models.yolo import Detect, Model
from models.experimental import attempt_download
from val import process_batch


class Dataset:
    """Creating Dataset class for getting Image and labels"""

    def __init__(self, dataloader, batch_s):
        self.dataloader = dataloader
        self.batch_size = batch_s
        self.im = None
        for _, (img, target, _, _) in enumerate(self.dataloader):
            self.im = img.to('cpu')
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

    def eval_func(self, model):
        """ eval_func """

        training = False
        single_cls = False
        nc = 4  # number of classes #if single_cls else int(data['nc'])
        iouv = torch.linspace(0.5, 0.95, 10, device='cpu')  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        augment = False
        save_hybrid = False
        save_dir = Path('')
        plots = False
        verbose = False
        callbacks = Callbacks()
        device = 'cpu'

        # conda
        task = 'val'
        conf_thres = 0.001  # confidence threshold
        iou_thres = 0.6  # NMS IoU threshold
        max_det = 300
        compute_loss = None

        # Dataloader

        dataloader = self.dataloader

        seen = 0
        names = model.names if hasattr(model, 'names') else model.module.names  # get class names

        if isinstance(names, (list, tuple)):  # old format
            names = dict(enumerate(names))

        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        _, _, p, r, _, mp, mr, map50, ap50, mAp = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        dt = Profile(), Profile(), Profile()  # profiling times
        _, stats, ap, ap_class = [], [], [], []
        callbacks.run('on_val_start')
        pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            callbacks.run('on_val_batch_start')
            with dt[0]:
                im = im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                nb, _, height, width = im.shape  # batch size, channels, height, width

            # Inference
            with dt[1]:
                preds, _ = model(im) if compute_loss else (model(im, augment=augment), None)

            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels

            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            with dt[2]:
                preds = non_max_suppression(preds,
                                            conf_thres,
                                            iou_thres,
                                            labels=lb,
                                            multi_label=True,
                                            agnostic=single_cls,
                                            max_det=max_det)

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

                # Predictions
                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

        if len(stats) and stats[0].any():
            _, _, p, r, _, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, mAp = p.mean(), r.mean(), ap50.mean(), ap.mean()

        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
        # Print results
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, mAp))
        if nt.sum() == 0:
            LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

        # Print results per class
        if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        return mAp


def main():
    """ Main """

    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--outpath',
                        type=str,
                        required=False,
                        default='./inc_compressed_model/output',
                        help='absolute path to save quantized model. By default it '
                             'will be saved in "./inc_compressed_model/output" folder')
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
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=False,
                        default="yolov5s.pt",
                        help='Model Weights ".pt" format')

    # Command line Arguments
    FLAGS = parser.parse_args()
    config_path = FLAGS.config
    out_path = FLAGS.outpath
    data_yml = FLAGS.data_yaml

    data = check_yaml(data_yml)  # check YAML
    data = check_dataset(data)  # check
    single_cls = False
    task = 'val'
    batch_size = 1

    # Load model
    device = 'cpu'  # Device
    weights = FLAGS.weights

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    ckpt = torch.load(attempt_download(w), map_location='cpu')
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()
    stride = 32

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

    imgsz = check_img_size(640, s=stride)  # check image size
    model.eval()

    dataloader_l = create_dataloader(data[task],
                                     imgsz,
                                     batch_size,
                                     stride,
                                     single_cls=single_cls,
                                     pad=0.5,
                                     rect=False,  # pt,
                                     workers=8,
                                     prefix=colorstr(f'{task}: '))[0]

    # Quantization
    quantizer = Quantization(config_path)
    quantizer.model = model
    dataset = Dataset(dataloader_l, batch_size)
    quantizer.calib_dataloader = common.DataLoader(dataset)
    quantizer.eval_func = dataset.eval_func
    q_model = quantizer.fit()
    q_model.save(out_path)

    print("*" * 30)
    print("Succesfully Quantized model and saved at :", out_path)
    print("*" * 30)


# Define the command line arguments to input the Hyperparameters - batchsize & Learning Rate
if __name__ == "__main__":
    main()
