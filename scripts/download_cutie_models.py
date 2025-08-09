#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import torch
import logging
import threading
import numpy as np
from torchvision.transforms.functional import to_tensor

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

if __name__ == "__main__":
    cutie = get_default_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    print("initalize complete!")
