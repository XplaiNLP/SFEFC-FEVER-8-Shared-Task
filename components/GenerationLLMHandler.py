from llama_cpp import (
    Llama,
)  # install with CUDA support: export CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
import torch
from typing import List
from logger import logger

import os
os.environ["HF_HOME"] = "./models"

class GenerationLLMHandler:
    def __init__(
        self,
        # model: str = "bartowski/Llama-3.2-3B-Instruct-GGUF",
        # model: str = "unsloth/Phi-4-mini-instruct-GGUF",
        model: str = "unsloth/phi-4-GGUF",
        filename_pattern: str = "*6_K.gguf",
        n_ctx: int = 10000,
        **llm_kwargs,
    ):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.llm = Llama.from_pretrained(
            repo_id=model,
            filename=filename_pattern,
            n_ctx=n_ctx,
            n_gpu_layers=(-1 if self.device == "cuda" else 0),
            verbose=False,
            **llm_kwargs,
        )

    def generate_question(
        self,
        claim: str,
    ) -> List[str]:
        
        system_prompt = "You are a professional fact checker. You recieve a claim from the user. Please provide a question you would ask to find out if a given claim is true, or not. Generate only one single question!"
        user_content = f"The claim you need to check: {claim}\nYour Question:\n"

        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        content = resp["choices"][0]["message"]["content"]
        logger.debug(f"Generated question: {content}")
        return content

    def generate_prediction(
        self,
        claim: str,
        retrieved_evidences: str,
    ) -> str:
        
        system_prompt = "You are a professional fact checker. You get a claim and provided evidence. Assess if the claim is supported or refuted by the evidence! Return only the result, either 'Supported' or 'Refuted'. "
        user_content = f"The claim: {claim} \n The evidence: {retrieved_evidences} \n Your verdict: "

        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        content = resp["choices"][0]["message"]["content"]
        content = content.strip()
        logger.debug(f"Generated prediction: {content}")
        return content