import numpy as np


class VeracityPredictor:
    def __init__(self):
        pass

    def get_veracity_prediction_from_llm_output(self, llm_prediction_output):
        pred_label = "Not Enough Evidence"
        if "supported" in llm_prediction_output.lower():
            pred_label = "Supported"
        if "refuted" in llm_prediction_output.lower():
            pred_label = "Refuted"
        return pred_label

    def get_veracity_prediction_by_semantic_filtering(self, similarities):
        threshold_ne_min_sim = 0.82
        threshold_c_var_sim = 0.0007
        pred = None
        if similarities[0] < threshold_ne_min_sim:
            pred = "Not Enough Evidence"
        if np.var(similarities) > threshold_c_var_sim:
            pred = "Conflicting Evidence/Cherrypicking"
        return pred
