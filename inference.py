import os
import random
import numpy as np
import torch
import argparse
import time
from logger import logger

from components.EvidenceRetriever import EvidenceRetriever
from components.EmbeddingLLMHandler import EmbeddingLLMHandler
from components.GenerationLLMHandler import GenerationLLMHandler
from components.ResultsHandler import ResultsHandler
from components.VeracityPredictor import VeracityPredictor


def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SEED = 0
set_all_seeds(SEED)


def main(args):
    start_time = time.time()

    gen_llm = GenerationLLMHandler(seed=SEED)
    emb_llm = EmbeddingLLMHandler(model_name="thenlper/gte-base")
    evidence_retriever = EvidenceRetriever(
        emb_llm=emb_llm, knowledge_store_dir_path=f"{args.knowledge_store_dir}"
    )
    veracity_predictor = VeracityPredictor()

    label_set = f"{args.target_data}"
    results_handler = ResultsHandler(label_set)
    
    start_time_processing_only = time.time()
    times_per_claim = []
    for i, _ in enumerate(results_handler.claims_data):
        try:
            start_time_times_per_claim = time.time()
            claim = results_handler.claims_data[i]["claim"]
            claim_id = results_handler.claims_data[i]["claim_id"]

            q = gen_llm.generate_question(claim)
            results_handler.generated_questions.append(q)

            retrieved_evidences, urls, similarities = (
                evidence_retriever.retrieve_evidences(claim + " " + q, claim_id)
            )

            semantic_filter_pred = (
                veracity_predictor.get_veracity_prediction_by_semantic_filtering(
                    similarities
                )
            )
            if semantic_filter_pred is not None:
                pred_label = semantic_filter_pred
                logger.debug(f"Veracity prediction via Semantic Filtering: {pred_label}")
            else:
                pred = gen_llm.generate_prediction(claim, retrieved_evidences)
                pred_label = veracity_predictor.get_veracity_prediction_from_llm_output(
                    pred
                )

            results_handler.add_prediction(q, retrieved_evidences, urls, pred_label)
            
            end_time_times_per_claim = time.time()
            time_delta_seconds = round(
                end_time_times_per_claim - start_time_times_per_claim, 2
            )
            times_per_claim.append(time_delta_seconds)
            logger.info(f"Claim processed in: {time_delta_seconds} s")
        except Exception as e:
            logger.error(f"Claim {i}: {e}")

    results_path = results_handler.save_results(args.json_output)
    logger.info(f"Results saved in: {results_path}")

    end_time = time.time()
    time_delta_seconds = round(end_time - start_time, 2)
    time_delta_seconds_processing_only = round(end_time - start_time_processing_only, 2)
    logger.info(f"Total running time in (seconds): {time_delta_seconds}")
    logger.info(
        f"Processing time only time delta (seconds): {time_delta_seconds_processing_only}; Processing time only per claim: {np.mean(times_per_claim)}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--target_data", default="data_store/averitec/dev.json")
    parser.add_argument("-o", "--json_output", default="preds.json")
    parser.add_argument("-k", "--knowledge_store_dir", default="knowledge_store/dev/")
    args = parser.parse_args()
    main(args)
