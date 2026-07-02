from __future__ import annotations

import logging

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from utils.langfuse import get_langfuse_client, start_observation

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Scores a RAG pipeline's retrieval and generation quality using RAGAS metrics
    (faithfulness, answer_relevancy, context_precision, context_recall), judged by
    a caller-supplied LLM and embeddings model.

    Args:
        judge_llm: A ragas-compatible judge LLM, e.g. LangchainLLMWrapper(ChatOpenAI(...))
            or LangchainLLMWrapper(ChatGoogleGenerativeAI(...)). Provider-agnostic by
            design - this pipeline doesn't know or care which model backs it.
        judge_embeddings: A ragas-compatible embeddings model used by metrics that need
            semantic similarity (e.g. answer_relevancy, context_precision).
    """

    def __init__(self, judge_llm, judge_embeddings) -> None:
        self._validate_config(judge_llm, judge_embeddings)
        self.judge_llm = judge_llm
        self.judge_embeddings = judge_embeddings
        self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    def run(self, dataset: EvaluationDataset) -> EvaluationResult:
        """
        Runs RAGAS evaluation over the given dataset using the configured judge.

        Args:
            dataset: A ragas EvaluationDataset already populated with
                user_input / response / retrieved_contexts / reference per sample.

        Returns:
            EvaluationResult - use .to_pandas() for per-sample scores and aggregate means.

        Raises:
            ValueError: If the dataset is empty.
            RuntimeError: If the RAGAS evaluation call fails (e.g. judge API errors).
        """
        if len(dataset) == 0:
            raise ValueError("Cannot evaluate an empty EvaluationDataset.")

        langfuse = get_langfuse_client()
        with start_observation(
            langfuse,
            name="evaluation_pipeline.run",
            as_type="span",
            input={
                "row_count": len(dataset),
                "metrics": [m.name for m in self.metrics],
            },
        ) as eval_span:
            try:
                result = evaluate(
                    dataset=dataset,
                    metrics=self.metrics,
                    llm=self.judge_llm,
                    embeddings=self.judge_embeddings,
                )
                if eval_span is not None:
                    eval_span.update(output={"scores": str(result)})
                return result
            except Exception as e:
                raise RuntimeError(f"Evaluation pipeline failed: {str(e)}") from e

    def _validate_config(self, judge_llm, judge_embeddings) -> None:
        """Validates constructor inputs before running any evaluation."""
        if judge_llm is None:
            raise ValueError("judge_llm must be provided.")
        if judge_embeddings is None:
            raise ValueError("judge_embeddings must be provided.")
