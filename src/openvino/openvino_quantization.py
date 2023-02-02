# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Openvino quantization
"""
# pylint: disable=E0401, C0301, E1101, C0103, R0914, E0611, C0301, C0413, C1801, E0402

import sys
sys.path.insert(0, '../yolov5')
from utils.dataloaders import create_dataloader
from utils.general import (check_dataset, check_img_size, check_yaml,
                           colorstr, non_max_suppression, xywh2xyxy)
from utils.metrics import ap_per_class
from val import process_batch
import copy
import os
from pathlib import Path
import argparse
import torch
import torch.nn.functional as nnf
from addict import Dict
from compression.api import Metric
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline


class Accuracy(Metric):  # define a class Accuracy
    """ Defining clas """
    mean_AP = 0

    def __init__(self, cls_names):
        super().__init__()
        self._name = "Mean_AP"
        self._matches = []
        self.names = cls_names
        self.stats_c = []

    @property
    def value(self):
        """Returns accuracy metric value for the last model output."""
        return {self._name: Accuracy.mean_AP}

    @property
    def avg_value(self):
        """
        Returns accuracy metric value for all model outputs. Results per image are stored in
        self._matches, where True means a correct prediction and False a wrong prediction.
        Accuracy is computed as the number of correct predictions divided by the total
        number of predictions.
        """
        # Compute metrics
        plots = False
        save_dir = Path('')
        stats = self.stats_c
        mAp = 0.0
        class_names = self.names
        if isinstance(class_names, (list, tuple)):  # old format
            class_names = dict(enumerate(class_names))
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            # import pdb;pdb.set_trace()
            _, _, _, _, _, ap, _ = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=class_names)
            ap = ap.mean(1)  # AP@0.5:0.95
            mAp = ap.mean()

        Accuracy.mean_AP = mAp
        return {self._name: Accuracy.mean_AP}

    def update(self, output, target):
        """Updates prediction matches.

        :param output: model output
        :param target: annotations
        """
        iouv = torch.linspace(0.5, 0.95, 10, device='cpu')  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        device = 'cpu'

        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IoU threshold
        max_det = 1000
        seen = 0

        elem1 = torch.tensor(output[1])
        elem2 = torch.tensor(output[2])
        elem3 = torch.tensor(output[3])

        pred_q = (torch.tensor(output[0]), [elem1, elem2, elem3])
        targets = target[0]
        targets[:, 2:] *= torch.tensor((640, 640, 640, 640), device=device)

        lb = []
        preds = non_max_suppression(pred_q, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False,
                                    max_det=max_det)
        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    self.stats_c.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            predn = pred.clone()

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            self.stats_c.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    def reset(self):
        """
        Resets the Accuracy metric. This is a required method that should initialize all
        attributes to their initial value.
        """
        self._matches = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {"direction": "higher-better", "type": "Mean_AP"}}


class Dataset:
    """Creating Dataset class for getting Image and labels"""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.im = None
        self.targets = None

    def __getitem__(self, index):
        (img, targets, _, _) = next(iter(self.dataloader))
        im = ((img.to('cpu').float()) / 255)
        im = nnf.interpolate(im, size=(640, 640), mode='bicubic', align_corners=False)
        self.im = im
        self.targets = targets
        return self.im, self.targets

    def __len__(self):
        return len(self.dataloader)


if __name__ == "__main__":
    # ## Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--FPIR_modelpath',
                        type=str,
                        required=False,
                        default='./openvino/openvino_models/openvino_ir/',
                        help='FP32 IR Model absolute path without extension')
    parser.add_argument('-o',
                        '--outpath',
                        type=str,
                        required=False,
                        default='./openvino/openvino_models/openvino_quantized',
                        help='default output quantized model will be save in path specified by outpath')
    parser.add_argument('-d',
                        '--data_yaml',
                        type=str,
                        required=False,
                        default='./data/VOC.yaml',
                        help='Absolute path to the yaml file containing paths data/ download data')
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=1,
                        help='batch size used for loading the data')
    FLAGS = parser.parse_args()
    model_path = FLAGS.FPIR_modelpath
    out_path = FLAGS.outpath
    data_path = FLAGS.data_yaml
    batch_size = FLAGS.batchsize

    model_config = Dict(
        {
            "model_name": "TrafficOD_Onnx_Model",
            "model": model_path + "TrafficOD_Onnx_Model" + ".xml",
            "weights": model_path + "TrafficOD_Onnx_Model" + ".bin",
        }
    )

    engine_config = Dict({"device": "CPU", "stat_requests_number": 2, "eval_requests_number": 2})

    algorithms = [
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "CPU",
                "preset": "performance",
                "stat_subset_size": 1000
            }
        }
    ]

    # Step 1: Load the model

    model = load_model(model_config=model_config)

    original_model = copy.deepcopy(model)

    # Step 2: Initialize the data loader

    data = check_yaml(data_path)
    data = check_dataset(data)  # check
    task = 'val'

    imgsz = check_img_size(640, s=32)  # stride)  # check image size
    names = data['names']

    dataloader_l = create_dataloader(data[task],
                                     imgsz,
                                     batch_size,
                                     32,
                                     single_cls=False,
                                     pad=0.5,
                                     # rect=pt,
                                     rect=False,
                                     workers=8,
                                     prefix=colorstr(f'{task}: '))[0]

    data_loader = Dataset(dataloader_l)

    print("This will take time, Please wait")
    # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric
    metric = Accuracy(names)

    # Step 4: Initialize the engine for metric calculation and statistics collection
    engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)

    # Step 5: Create a pipeline of compression algorithms
    pipeline = create_pipeline(algo_config=algorithms, engine=engine)

    original_metric_results = pipeline.evaluate(original_model)

    # Step 6: Execute the pipeline
    compressed_model = pipeline.run(model=model)

    # Step 7 (Optional): Compress model weights quantized precision
    #                    in order to reduce the size of final .bin file
    compress_model_weights(model=compressed_model)

    # Step 8: Save the compressed model and get the path to the model
    compressed_model_paths = save_model(
        model=compressed_model, save_path=os.path.join(os.path.curdir, out_path)
    )
    compressed_model_xml = Path(compressed_model_paths[0]["model"])
    print(f"The quantized model is stored in {compressed_model_xml}")

    # Step 9 (Optional): Evaluate the original and compressed model. Print the results

    quantized_metric_results = pipeline.evaluate(compressed_model)
    if quantized_metric_results:
        print(f"MeanAP of the quantized model: {next(iter(quantized_metric_results.values())):.5f}")

    if original_metric_results:
        print(f"MeanAP of the original model:  {next(iter(original_metric_results.values())):.5f}")
