import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingLLMHandler:
    def __init__(self, model_name="thenlper/gte-base"):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name
        if model_name == "BAAI/bge-m3":
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name).to(self.device)
            self.embedding_model.eval()
        else:
            self.embedding_model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text):
        if (
            "sentence-transformers" in self.model_name
            or self.model_name == "thenlper/gte-base"
        ):
            embedding = self.encode_with_sentence_transformers(text)
        if self.model_name == "BAAI/bge-m3":
            embedding = self.encode_with_bge_m3(text)
        return embedding

    def encode_with_sentence_transformers(self, text):
        embedding = self.embedding_model.encode(text, batch_size=32)
        return embedding

    def encode_with_bge_m3(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
