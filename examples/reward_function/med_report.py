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
from typing import Dict, List, Union
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge


def format_reward(predict: str) -> float:
    """检查预测输出是否符合格式要求
    
    Args:
        predict: 模型预测的输出
        
    Returns:
        float: 格式正确返回1.0，否则返回0.0
    """
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def extract_answer(predict: str) -> str:
    """从预测输出中提取答案部分
    
    Args:
        predict: 模型预测的输出
        
    Returns:
        str: 提取的答案内容
    """
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
        return content_match.group(1).strip() if content_match else predict.strip()
    except Exception:
        return predict.strip()


def bleu_reward(predict: str, ground_truth: str) -> float:
    """计算BLEU分数作为奖励
    
    Args:
        predict: 模型预测的输出
        ground_truth: 真实标签
        
    Returns:
        float: BLEU分数，范围[0, 1]
    """
    try:
        predict_text = extract_answer(predict)
        predict_tokens = word_tokenize(predict_text.lower())
        ground_truth_tokens = word_tokenize(ground_truth.lower())
        
        # 计算BLEU分数，使用1-4gram权重
        weights = (0.25, 0.25, 0.25, 0.25)
        return sentence_bleu([ground_truth_tokens], predict_tokens, weights=weights)
    except Exception:
        return 0.0


def rouge_reward(predict: str, ground_truth: str) -> Dict[str, float]:
    """计算ROUGE分数作为奖励
    
    Args:
        predict: 模型预测的输出
        ground_truth: 真实标签
        
    Returns:
        Dict[str, float]: 包含ROUGE-1, ROUGE-2, ROUGE-L的F1分数
    """
    try:
        rouge = Rouge()
        predict_text = extract_answer(predict)
        scores = rouge.get_scores(predict_text, ground_truth)[0]
        
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }
    except Exception:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}


def compute_score(predict: Union[str, List[str]], ground_truth: Union[str, List[str]], 
                 format_weight: float = 0.2, bleu_weight: float = 0.4, rouge_weight: float = 0.4) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """计算总体评分
    
    Args:
        predict: 模型预测的输出，可以是单个字符串或字符串列表
        ground_truth: 真实标签，可以是单个字符串或字符串列表
        format_weight: 格式评分权重
        bleu_weight: BLEU评分权重
        rouge_weight: ROUGE评分权重
        
    Returns:
        Union[Dict[str, float], List[Dict[str, float]]]: 评分结果
    """
    # 处理单个预测和真实标签的情况
    if isinstance(predict, str) and isinstance(ground_truth, str):
        return _compute_single_score(predict, ground_truth, format_weight, bleu_weight, rouge_weight)
    
    # 处理多个预测和真实标签的情况
    if isinstance(predict, list) and isinstance(ground_truth, list):
        scores = []
        for pred, truth in zip(predict, ground_truth):
            # 处理格式问题，移除多余空格
            pred = re.sub(r"\s*(<|>|/)\s*", r"\1", pred)
            scores.append(_compute_single_score(pred, truth, format_weight, bleu_weight, rouge_weight))
        return scores
    
    raise ValueError("predict和ground_truth类型必须匹配，且都为str或List[str]")


def _compute_single_score(predict: str, ground_truth: str, format_weight: float, 
                          bleu_weight: float, rouge_weight: float) -> Dict[str, float]:
    """计算单个预测和真实标签的评分
    
    Args:
        predict: 模型预测的输出
        ground_truth: 真实标签
        format_weight: 格式评分权重
        bleu_weight: BLEU评分权重
        rouge_weight: ROUGE评分权重
        
    Returns:
        Dict[str, float]: 评分结果
    """
    format_score = format_reward(predict)
    bleu_score = bleu_reward(predict, ground_truth)
    
    rouge_scores = rouge_reward(predict, ground_truth)
    # 使用ROUGE-L作为主要ROUGE指标
    rouge_score = rouge_scores["rouge-l"]
    
    # 计算总体评分
    overall_score = (format_weight * format_score + 
                    bleu_weight * bleu_score + 
                    rouge_weight * rouge_score)
    
    return {
        "overall": overall_score,
        "format": format_score,
        "bleu": bleu_score,
        "rouge": rouge_scores
    }