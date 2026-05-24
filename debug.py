print("1")

import torch
print("2 torch ok")

from utils import *
print("3 utils ok")

from models import *
print("4 models ok")

from utils.calibration_metrics import (
    collect_logits_and_labels,
    compute_accuracy
)
print("5 metrics ok")

from models import load_resnet50
print("6 resnet ok")

print("DONE")