from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings


class EvaluationPipeline:
    """Evaluates RAG pipeline outputs using RAGAS metrics with a separate judge LLM."""

    def __init__(self, ollama_model_name: str, embedding_model_name: str):
        self.llm = LangchainLLMWrapper(ChatOllama(model=ollama_model_name, temperature=0, max_tokens=128))
        self.embedder = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embedding_model_name))
        self.metrics = [faithfulness, answer_relevancy, context_precision]

    def run(self, dataset: EvaluationDataset):
        """Run RAGAS evaluation on the given dataset and return the results."""
        return evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embedder,
        )