import numpy as np
import pandas as pd

targets = [
    ["Homo sapiens"],
    ["Homo sapiens"],
    ["Homo sapiens", "Pan troglodytes", "Gorilla gorilla"],
    ["Panthera leo", "Panthera pardus", "Acinonyx jubatus"],
    ["Zea mays", "Oryza sativa", "Triticum aestivum"],
    ["Quercus robur", "Picea abies", "Fagus sylvatica"],
    ["Escherichia coli", "Saccharomyces cerevisiae"],
    ["Carcharodon carcharias", "Delphinus delphis", "Balaenoptera musculus"],
    [],
    []
]

predictions = [
    ["Homo sapiens"], #juste
    ["Homo homo"], #faux
    ["Homo sapiens", "Pan troglodites", "Gorilla gorilla"], #1faux
    ["Panthera leo", "Panthera", "Acinonyx jubatu"], #2faux
    ["Zea mays", "Oryza sativa", "Triticum aestivum"], #juste
    ["Quercus robur", "Picea abies"], #1manquant
    ["Escherichia coli", "Saccharomyces cerevisiae"], #juste
    [], # 3manquants
    [], #juste
    ["La femme"] #faux
]

predictions_1_2 = [
    ["Homo sapiens"],
    ["Homo sapiens"],
    ["Homo sapiens", "Pan troglodytes", "Gorilla gorilla"],
    ["Panthera leo", "Panthera pardus", "Acinonyx jubatus"],
    ["Zea mays", "Oryza sativa", "Triticum aestivum"],
    ["Quercus robur", "Picea abies", "Fagus sylvatica"],
    [],
    ["Carcharodon carcharias", "Delphinus delphis", "Balaenoptera musculus"],
    [],
    []
] # 1 texte faux, 2 erreurs

predictions_1_1= [
    ["Homo sapiens"],
    ["Homo sapiens"],
    ["Homo sapiens", "Pan troglodytes", "Gorilla gorilla"],
    ["Panthera leo", "Panthera pardus", "Acinonyx jubatus"],
    ["Zea mays", "Oryza sativa", "Triticum aestivum"],
    ["Quercus robur", "Picea abies", "Fagus sylvatica"],
    ["Escherichia coli"],
    ["Carcharodon carcharias", "Delphinus delphis", "Balaenoptera musculus"],
    [],
    []
] # 1 texte faux, 1 erreur


class Evaluator:
    """

    """

    def __init__(self):
        pass
    
    def compute_tp_fp_fn(self, target, prediction):
        target_set = set(target)
        pred_set = set(prediction)
        tp = len(target_set & pred_set)
        fp = len(pred_set - target_set)
        fn = len(target_set - pred_set)
        return tp, fp, fn

    def calculate_metrics(self, targets, predictions):
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
            tp, fp, fn = self.compute_tp_fp_fn(target, prediction)
            TP_global += tp
            FP_global += fp
            FN_global += fn

        precision_global = TP_global / (TP_global + FP_global) if (TP_global + FP_global) > 0 else 0
        recall_global = TP_global / (TP_global + FN_global) if (TP_global + FN_global) > 0 else 0
        f1_global = (2 * precision_global * recall_global /
                    (precision_global + recall_global)) if (precision_global + recall_global) > 0 else 0
        
        return {'precision': precision_global, 'recall' : recall_global, 'f1-score' :f1_global}


evaluator = Evaluator()

print(evaluator.calculate_metrics(targets, targets))
print(evaluator.calculate_metrics(targets, predictions_1_1))
print(evaluator.calculate_metrics(targets, predictions_1_2))
