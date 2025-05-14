import json


class ResultsHandler:
    def __init__(self, path):
        self.claims_data = []
        self.veracity_preds = []
        self.total_predictions = 0
        self.generated_questions = []
        with open(path) as f:
            self.claims_data = json.load(f)

        for i, c in enumerate(self.claims_data):
            self.claims_data[i]["claim_id"] = i

    def add_prediction(self, q, retrieved_evidences, urls, pred_label):
        evidences = []
        for j, _ in enumerate(retrieved_evidences):
            question = q[j] if isinstance(q, list) else q
            evidence = {
                "question": question,
                "answer": retrieved_evidences[j],
                "url": urls[j],
            }
            evidences.append(evidence)

        pred = {
            "claim": self.claims_data[self.total_predictions]["claim"],
            "claim_id": self.claims_data[self.total_predictions]["claim_id"],
            "evidence": evidences,
            "pred_label": pred_label,
        }
        self.veracity_preds.append(pred)
        self.total_predictions += 1

    def save_results(self, json_output_path):
        with open(json_output_path, "w", encoding="utf-8") as output_json:
            output_json.write(
                json.dumps(self.veracity_preds, ensure_ascii=False) + "\n"
            )
        return json_output_path