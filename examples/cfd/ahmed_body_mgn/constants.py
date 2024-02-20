# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple, Optional


class Constants(BaseModel):
    """Ahmed Body model constants"""

    ckpt_path: str = "./checkpoints"
    ckpt_name: str = "./ahmed_body"
    # data_dir: str = "/workspace/gino_Jean/data/ahmed_modulus/"
    data_dir: str = "/workspace/gino_Jean/data/toyota/"
    results_dir: str = "./results"

    input_dim_nodes: int = 3
    input_dim_edges: int = 4
    output_dim: int = 4
    aggregation: int = "sum"
    hidden_dim_node_encoder: int = 12
    hidden_dim_edge_encoder: int = 12
    hidden_dim_node_decoder: int = 12
    hidden_dim_processor: int = 6

    batch_size: int = 1
    epochs: int = 500
    num_training_samples: int = 25
    num_validation_samples: int = 5
    num_test_samples: int = 2

    lr: float = 1e-4
    lr_decay_rate: float = 0.99985

    amp: bool = False
    jit: bool = False

    wandb_mode: str = "disabled"
