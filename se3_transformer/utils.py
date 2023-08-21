# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import argparse
from typing import Union, List, Dict

import torch
from torch import Tensor


def aggregate_residual(feats1, feats2, method: str):
    """ Add or concatenate two fiber features together. If degrees don't match, will use the ones of feats2. """
    if method in ['add', 'sum']:
        return {k: (v + feats1[k]) if k in feats1 else v for k, v in feats2.items()}
    elif method in ['cat', 'concat']:
        return {k: torch.cat([v, feats1[k]], dim=1) if k in feats1 else v for k, v in feats2.items()}
    else:
        raise ValueError('Method must be add/sum or cat/concat')


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


def unfuse_features(features: Tensor, degrees: List[int]) -> Dict[str, Tensor]:
    return dict(zip(map(str, degrees), features.split([degree_to_dim(deg) for deg in degrees], dim=-1)))


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def to_cuda(x, device: str = 'cuda'):
    """ Try to convert a Tensor, a collection of Tensors or a DGLGraph to CUDA """
    if not torch.cuda.is_available() or device != 'cuda':
        return x
    if isinstance(x, Tensor):
        return x.cuda(non_blocking=True)
    elif isinstance(x, tuple):
        return (to_cuda(v) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=torch.cuda.current_device(), non_blocking=True)
