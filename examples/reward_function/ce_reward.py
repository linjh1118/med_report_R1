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
from sklearn.metrics import precision_score, recall_score, f1_score

def ce_matrix(pred_labels, gt_labels):
    pred_labels_name = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
    gt_labels_name = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"]
    pred_labels_cal = [0] * len(gt_labels_name)
    for pred_idx, label in pred_labels:
        gt_idx = gt_labels_name.index(pred_labels_name[pred_idx])
        pred_labels_cal[gt_idx] = label
    pred_labels_cal = [1 if label == 1 else 0 for label in pred_labels_cal]
    return {
        'p': precision_score(gt_labels, pred_labels_cal, average='macro'),
        'r': recall_score(gt_labels, pred_labels_cal, average='macro'),
        'f': f1_score(gt_labels, pred_labels_cal, average='macro')
    }

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
    pred_labels = response.json()["response"][0]
    ground_truth = json.loads(ground_truth)
    gt_labels = ground_truth['label_vec']
    return ce_matrix(pred_labels, gt_labels)['f']


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
