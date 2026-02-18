import numpy as np
import pandas as pd

    
# def compute_tp_fp_fn(self, target, prediction):
#     target_set = set(target)
#     pred_set = set(prediction)
#     tp = len(target_set & pred_set)
#     fp = len(pred_set - target_set)
#     fn = len(target_set - pred_set)
#     return tp, fp, fn

def calculate_metrics_global(predictions, targets):
    """
    Args:
        targets: True target values
        predictions: Extracted values
        
    Returns:
        Dictionary with calculated metrics
    """
    
    TP_global = 0
    FP_global = 0
    FN_global = 0

    for target, prediction in zip(targets, predictions):
        target_set = set(target)
        pred_set = set(prediction)
        TP_global += len(target_set & pred_set)
        FP_global += len(pred_set - target_set)
        FN_global += len(target_set - pred_set)

    precision_global = TP_global / (TP_global + FP_global) if (TP_global + FP_global) > 0 else 0
    recall_global = TP_global / (TP_global + FN_global) if (TP_global + FN_global) > 0 else 0
    f1_global = (2 * precision_global * recall_global /
                (precision_global + recall_global)) if (precision_global + recall_global) > 0 else 0
    
    return precision_global, recall_global, f1_global

def calculate_metrics(extracted, ground_truth) :
    if not extracted and not ground_truth :
        return 1.0, 1.0, 1.0
    if not extracted or not ground_truth :
        return 0.0, 0.0, 0.0
    
    correct = len(extracted.intersection(ground_truth))
    precision = correct / len(extracted)
    recall = correct / len(ground_truth)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

