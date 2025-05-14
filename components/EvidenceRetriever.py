# Note: Parts of sparse retrieval via BM25 are reused from https://github.com/Raldir/FEVER-8-Shared-Task/blob/main/baseline/retrieval_optimized.py
from logger import logger
from rank_bm25 import BM25Okapi
import numpy as np
import json

import nltk
nltk.download("punkt_tab")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class EvidenceRetriever:
    def __init__(self, emb_llm, knowledge_store_dir_path="knowledge_store/dev/"):
        self.knolwedge_store_dir_path = knowledge_store_dir_path
        self.bm25_top_k = 1500
        self.cos_sim_top_k = 10
        self.embedding_model = emb_llm
        self.concatenate_sentences = True
        self.concat_sentence_amount = 4

    def retrieve_evidences(self, claim, claim_id):
        bm25_top_k_sentences, bm25_urls = self.get_bm25_results_for_claim(
            claim, claim_id
        )
        evidences, urls, similarities = self.get_top_k_cos_sim_results_for_claim(
            claim, bm25_top_k_sentences, bm25_urls
        )
        return evidences, urls, similarities

    def get_bm25_results_for_claim(self, claim, claim_id):
        knowledge_store_file_path = f"{self.knolwedge_store_dir_path}{claim_id}.json"
        sentences = []
        urls = []
        if self.concatenate_sentences:
            with open(knowledge_store_file_path, "r", encoding="utf-8") as json_file:
                for j, line in enumerate(json_file):
                    data = json.loads(line)
                    url2text = data["url2text"]
                    url = data["url"]

                    for i in range(0, len(url2text), self.concat_sentence_amount):
                        chunk = url2text[i : i + self.concat_sentence_amount]
                        concatenated_sentence = " ".join(chunk)
                        sentences.append(concatenated_sentence)
                        urls.append(url)
        else:
            with open(knowledge_store_file_path, "r", encoding="utf-8") as json_file:
                for j, line in enumerate(json_file):
                    data = json.loads(line)
                    sentences.extend(data["url2text"])
                    urls.extend([data["url"] for _ in range(len(data["url2text"]))])

        bm25_top_k_sentences, urls = self.retrieve_top_k_sentences_with_bm25(
            claim, sentences, urls, top_k=self.bm25_top_k
        )
        return bm25_top_k_sentences, urls

    def retrieve_top_k_sentences_with_bm25(self, query, document, urls, top_k):
        tokenized_docs = [nltk.word_tokenize(doc) for doc in document[:top_k]]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(nltk.word_tokenize(query))
        top_k_idx = np.argsort(scores)[::-1][:top_k]
        return [document[i] for i in top_k_idx], [urls[i] for i in top_k_idx]

    def get_top_k_cos_sim_results_for_claim(self, claim, bm25_top_k_sentences, urls):
        cos_sim_top_k_sentences = self.retrieve_top_k_sentences_with_cos_sim(
            claim, bm25_top_k_sentences, urls
        )
        retrieved_sentences = [s[0] + "\n" for s in cos_sim_top_k_sentences]
        similarities = [s[1] for s in cos_sim_top_k_sentences]
        urls = [s[2] + "\n" for s in cos_sim_top_k_sentences]
        return retrieved_sentences, urls, similarities

    def retrieve_top_k_sentences_with_cos_sim(self, query, sentences, urls):
        query_embedding = self.embedding_model.encode(query)
        sentence_embeddings = np.array(self.embedding_model.encode(sentences))
        similarities = [
            cosine_similarity(query_embedding, sentence_embeddings[i])
            for i in range(len(sentence_embeddings))
        ]
        sentence_similarity_pairs = list(zip(sentences, similarities, urls))
        sorted_pairs = sorted(
            sentence_similarity_pairs, key=lambda x: x[1], reverse=True
        )
        return sorted_pairs[: self.cos_sim_top_k]