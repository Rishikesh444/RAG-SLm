import re
from typing import List, Dict, Any, Optional

class RAGEvaluator:
    """
    RAG Quality Evaluation Engine measuring Faithfulness, Answer Relevance,
    Context Relevance, and Context Recall.
    """

    def tokenize(self, text: str) -> set[str]:
        return set(re.findall(r'\w+', text.lower()))

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline output quality across 4 core metrics.
        
        Returns:
            Dictionary of metrics with values between 0.0 and 1.0.
        """
        context_text = " ".join([c.get("text", "") for c in retrieved_chunks])
        
        q_tokens = self.tokenize(question)
        ans_tokens = self.tokenize(answer)
        ctx_tokens = self.tokenize(context_text)

        # 1. Faithfulness: How much of the generated answer is grounded in retrieved context?
        faithfulness = self._calculate_overlap(ans_tokens, ctx_tokens)

        # 2. Answer Relevance: How well does the answer address the question?
        answer_relevance = self._calculate_overlap(q_tokens, ans_tokens)

        # 3. Context Relevance: How much of the retrieved context is relevant to the question?
        context_relevance = self._calculate_overlap(q_tokens, ctx_tokens)

        # 4. Context Recall: (If ground truth provided) How much ground truth is in retrieved context?
        context_recall = None
        if ground_truth:
            gt_tokens = self.tokenize(ground_truth)
            context_recall = self._calculate_overlap(gt_tokens, ctx_tokens)

        return {
            "faithfulness": round(faithfulness, 4),
            "answer_relevance": round(answer_relevance, 4),
            "context_relevance": round(context_relevance, 4),
            "context_recall": round(context_recall, 4) if context_recall is not None else "N/A",
            "overall_score": round(
                (faithfulness + answer_relevance + context_relevance) / 3.0, 4
            )
        }

    def _calculate_overlap(self, set_a: set[str], set_b: set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = set_a.intersection(set_b)
        return len(intersection) / len(set_a)

evaluator = RAGEvaluator()
