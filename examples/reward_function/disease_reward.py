# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from typing import Dict, List
import requests
import json
from sklearn.metrics import f1_score


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think><answer>[^\n]*</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0

def accuracy_reward(predict: str, ground_truth: str) -> float:
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, predict, re.DOTALL)
    if match:
        predict = match.group(1)
    else:
        predict = ""
    response = requests.post("http://localhost:8002/get_response", json={"sentences": [predict]})
    labels = response.json()["response"][0]
    labels_name = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
    ground_truth = json.loads(ground_truth)
    disease = ground_truth["disease"]
    gt_labels = [1 if label == disease else 0 for label in labels_name]
    pred_labels = [1 if e == 1 else 0 for e in labels]
    return f1_score(gt_labels, pred_labels, average="macro")


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores
