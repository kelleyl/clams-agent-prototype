"""
RAG system for CLAMS evaluation data.
Provides retrieval-augmented generation capabilities based on historical
pipeline evaluations from the aapb-evaluations repository.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured representation of an evaluation result."""
    task_type: str  # ASR, TextRecognition, TimeFrameLabeling, etc.
    app_name: str
    app_version: str
    batch_id: str
    metrics: Dict[str, float]  # e.g., {"precision": 0.90, "recall": 0.83, "f1": 0.86}
    per_document_scores: Dict[str, float]  # GUID -> score
    evaluation_date: Optional[str]
    dataset_description: str
    limitations: List[str]
    report_path: str

    def to_text(self) -> str:
        """Convert to text for embedding."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
        return f"""
Task: {self.task_type}
App: {self.app_name} (version {self.app_version})
Batch: {self.batch_id}
Metrics: {metrics_str}
Dataset: {self.dataset_description}
Limitations: {'; '.join(self.limitations) if self.limitations else 'None noted'}
Number of documents evaluated: {len(self.per_document_scores)}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EvaluationDataIngester:
    """Ingests evaluation data from aapb-evaluations repository."""

    def __init__(self, evaluations_dir: str):
        self.evaluations_dir = Path(evaluations_dir)
        self.results: List[EvaluationResult] = []

    def ingest_all(self) -> List[EvaluationResult]:
        """Ingest all evaluation reports from the repository."""
        task_dirs = [
            "AutomaticSpeechRecognition",
            "TextRecognition",
            "TimeFrameLabeling",
            "TimePointLabeling",
            "NamedEntityRecognition",
            "NamedEntityGrounding",
            "ForcedAlignment",
            "KeyedInformationExtraction"
        ]

        for task_dir in task_dirs:
            task_path = self.evaluations_dir / task_dir
            if task_path.exists():
                self._ingest_task_dir(task_path, task_dir)

        logger.info(f"Ingested {len(self.results)} evaluation results")
        return self.results

    def _ingest_task_dir(self, task_path: Path, task_type: str):
        """Ingest all reports from a task directory."""
        for report_file in task_path.glob("report*.md"):
            try:
                result = self._parse_report(report_file, task_type)
                if result:
                    self.results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse {report_file}: {e}")

    def _parse_report(self, report_path: Path, task_type: str) -> Optional[EvaluationResult]:
        """Parse a markdown report file."""
        content = report_path.read_text()

        # Extract app name and version from filename or content
        app_name, app_version, batch_id = self._extract_app_info(report_path.name, content)

        # Extract metrics based on task type
        metrics = self._extract_metrics(content, task_type)

        # Extract per-document scores
        per_doc_scores = self._extract_per_document_scores(content, task_type)

        # Extract limitations
        limitations = self._extract_limitations(content)

        # Extract dataset description
        dataset_desc = self._extract_dataset_description(content)

        # Extract date from filename if present
        eval_date = self._extract_date(report_path.name)

        if not metrics:
            logger.debug(f"No metrics found in {report_path}")
            return None

        return EvaluationResult(
            task_type=task_type,
            app_name=app_name,
            app_version=app_version,
            batch_id=batch_id,
            metrics=metrics,
            per_document_scores=per_doc_scores,
            evaluation_date=eval_date,
            dataset_description=dataset_desc,
            limitations=limitations,
            report_path=str(report_path)
        )

    def _extract_app_info(self, filename: str, content: str) -> tuple:
        """Extract app name, version, and batch ID."""
        # Try to extract from filename patterns like:
        # report-20230725-preds@slatedetection@aapb-collaboration-7.md
        # report@parseqocr-wrapper1.0@batch2.md

        app_name = "unknown"
        app_version = "unknown"
        batch_id = "unknown"

        # Pattern 1: report-DATE-preds@APP@BATCH.md
        match = re.search(r'preds@([^@]+)@([^.]+)', filename)
        if match:
            app_name = match.group(1)
            batch_id = match.group(2)

        # Pattern 2: report@APP-VERSION@BATCH.md
        match = re.search(r'report@([^@]+)@([^.]+)', filename)
        if match:
            app_part = match.group(1)
            batch_id = match.group(2)
            # Try to split version from app name (e.g., parseqocr-wrapper1.0)
            version_match = re.search(r'(.+?)(\d+\.\d+)$', app_part)
            if version_match:
                app_name = version_match.group(1).rstrip('-')
                app_version = version_match.group(2)
            else:
                app_name = app_part

        # Try to extract version from content
        version_match = re.search(r'\[.+?\s+v(\d+(?:\.\d+)?)\]', content)
        if version_match and app_version == "unknown":
            app_version = version_match.group(1)

        return app_name, app_version, batch_id

    def _extract_metrics(self, content: str, task_type: str) -> Dict[str, float]:
        """Extract metrics based on task type."""
        metrics = {}

        if task_type in ["TimeFrameLabeling", "TimePointLabeling"]:
            # Look for Precision, Recall, F1
            precision_match = re.search(r'[Pp]recision\s*[=:]\s*([\d.]+)', content)
            recall_match = re.search(r'[Rr]ecall\s*[=:]\s*([\d.]+)', content)
            f1_match = re.search(r'F1\s*[=:]\s*([\d.]+)', content)

            if precision_match:
                metrics['precision'] = float(precision_match.group(1))
            if recall_match:
                metrics['recall'] = float(recall_match.group(1))
            if f1_match:
                metrics['f1'] = float(f1_match.group(1))

        elif task_type == "TextRecognition":
            # Look for CER
            cer_match = re.search(r'[Tt]otal\s+[Mm]ean\s+CER[:\s]+([\d.]+)', content)
            if cer_match:
                metrics['mean_cer'] = float(cer_match.group(1))

        elif task_type == "AutomaticSpeechRecognition":
            # Look for WER (case sensitive and insensitive)
            # Look in tables for average
            cases_match = re.search(r'\|\s*CaseS\s*\|\s*CaseI\s*\|', content)
            if cases_match:
                # Find the row with averages
                avg_match = re.search(r'\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', content[cases_match.end():])
                if avg_match:
                    metrics['wer_case_sensitive'] = float(avg_match.group(1))
                    metrics['wer_case_insensitive'] = float(avg_match.group(2))

            # Also try single WER values
            wer_match = re.search(r'[Aa]vera?ge?\s+WER[:\s]+([\d.]+)', content)
            if wer_match and 'wer_case_sensitive' not in metrics:
                metrics['wer'] = float(wer_match.group(1))

        elif task_type == "NamedEntityRecognition":
            # Look for entity-level and token-level metrics
            entity_f1 = re.search(r'[Ee]ntity.*?F1[:\s]+([\d.]+)', content)
            token_f1 = re.search(r'[Tt]oken.*?F1[:\s]+([\d.]+)', content)

            if entity_f1:
                metrics['entity_f1'] = float(entity_f1.group(1))
            if token_f1:
                metrics['token_f1'] = float(token_f1.group(1))

        return metrics

    def _extract_per_document_scores(self, content: str, task_type: str) -> Dict[str, float]:
        """Extract per-document scores from report content."""
        scores = {}

        # Look for GUID patterns with scores
        # Pattern: cpb-aacip-XXX-YYYYY: SCORE or cpb-aacip-XXX-YYYYY | SCORE
        guid_pattern = r'(cpb-aacip-[\w-]+)[:\s|]+\s*([\d.]+)'

        for match in re.finditer(guid_pattern, content):
            guid = match.group(1)
            try:
                score = float(match.group(2))
                scores[guid] = score
            except ValueError:
                continue

        return scores

    def _extract_limitations(self, content: str) -> List[str]:
        """Extract limitations section from report."""
        limitations = []

        # Look for limitations section
        limit_match = re.search(r'#+\s*[Ll]imitations.*?\n(.*?)(?=\n#+|\Z)', content, re.DOTALL)
        if limit_match:
            limit_text = limit_match.group(1)
            # Extract numbered items
            for match in re.finditer(r'\d+\.\s*(.+?)(?=\n\d+\.|\n\n|\Z)', limit_text, re.DOTALL):
                limitations.append(match.group(1).strip())

        return limitations

    def _extract_dataset_description(self, content: str) -> str:
        """Extract dataset description from report."""
        # Look for evaluation dataset section
        dataset_match = re.search(r'#+\s*[Ee]valuation\s+[Dd]ataset.*?\n(.*?)(?=\n#+|\Z)', content, re.DOTALL)
        if dataset_match:
            return dataset_match.group(1).strip()[:500]  # Limit length
        return "Not specified"

    def _extract_date(self, filename: str) -> Optional[str]:
        """Extract evaluation date from filename."""
        date_match = re.search(r'(\d{8})', filename)
        if date_match:
            date_str = date_match.group(1)
            try:
                return datetime.strptime(date_str, '%Y%m%d').isoformat()
            except ValueError:
                pass
        return None


class EvaluationVectorStore:
    """Vector store for evaluation results using simple TF-IDF based retrieval."""

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings = None
        self._vectorizer = None

    def add_results(self, results: List[EvaluationResult]):
        """Add evaluation results to the store."""
        for result in results:
            self.documents.append({
                'text': result.to_text(),
                'metadata': result.to_dict()
            })

        # Build TF-IDF index
        self._build_index()

    def _build_index(self):
        """Build TF-IDF index for retrieval."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            texts = [doc['text'] for doc in self.documents]
            self._vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000
            )
            self.embeddings = self._vectorizer.fit_transform(texts)
            logger.info(f"Built TF-IDF index with {len(texts)} documents")
        except ImportError:
            logger.warning("sklearn not available, using simple keyword matching")
            self._vectorizer = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant evaluation results."""
        if not self.documents:
            return []

        if self._vectorizer is not None and self.embeddings is not None:
            return self._tfidf_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)

    def _tfidf_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """TF-IDF based search."""
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.embeddings)[0]

        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                results.append({
                    'score': float(similarities[idx]),
                    'text': self.documents[idx]['text'],
                    'metadata': self.documents[idx]['metadata']
                })

        return results

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search fallback."""
        query_terms = query.lower().split()
        scored_docs = []

        for doc in self.documents:
            text_lower = doc['text'].lower()
            score = sum(1 for term in query_terms if term in text_lower)
            if score > 0:
                scored_docs.append({
                    'score': score,
                    'text': doc['text'],
                    'metadata': doc['metadata']
                })

        # Sort by score descending
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]

    def get_by_task(self, task_type: str) -> List[Dict[str, Any]]:
        """Get all results for a specific task type."""
        return [
            {'text': doc['text'], 'metadata': doc['metadata']}
            for doc in self.documents
            if doc['metadata'].get('task_type') == task_type
        ]

    def get_by_app(self, app_name: str) -> List[Dict[str, Any]]:
        """Get all results for a specific app."""
        return [
            {'text': doc['text'], 'metadata': doc['metadata']}
            for doc in self.documents
            if app_name.lower() in doc['metadata'].get('app_name', '').lower()
        ]

    def get_best_for_task(self, task_type: str, metric: str = None) -> Optional[Dict[str, Any]]:
        """Get the best performing app for a task type."""
        task_results = self.get_by_task(task_type)

        if not task_results:
            return None

        # Determine which metric to use based on task type
        if metric is None:
            metric_map = {
                'TimeFrameLabeling': 'f1',
                'TimePointLabeling': 'f1',
                'TextRecognition': 'mean_cer',
                'AutomaticSpeechRecognition': 'wer_case_insensitive',
                'NamedEntityRecognition': 'entity_f1'
            }
            metric = metric_map.get(task_type, 'f1')

        # For error rates (CER, WER), lower is better
        lower_is_better = metric in ['mean_cer', 'wer', 'wer_case_sensitive', 'wer_case_insensitive']

        best_result = None
        best_score = float('inf') if lower_is_better else float('-inf')

        for result in task_results:
            metrics = result['metadata'].get('metrics', {})
            score = metrics.get(metric)

            if score is not None:
                if lower_is_better:
                    if score < best_score:
                        best_score = score
                        best_result = result
                else:
                    if score > best_score:
                        best_score = score
                        best_result = result

        return best_result


class EvaluationRAG:
    """Main RAG interface for evaluation data."""

    def __init__(self, evaluations_dir: str = None):
        if evaluations_dir is None:
            # Default to sibling directory
            project_dir = Path(__file__).parent.parent.parent
            evaluations_dir = project_dir / "aapb-evaluations"

        self.evaluations_dir = Path(evaluations_dir)
        self.ingester = EvaluationDataIngester(self.evaluations_dir)
        self.vector_store = EvaluationVectorStore()
        self._initialized = False

    def initialize(self):
        """Initialize the RAG system by ingesting all evaluation data."""
        if self._initialized:
            return

        if not self.evaluations_dir.exists():
            logger.warning(f"Evaluations directory not found: {self.evaluations_dir}")
            self._initialized = True
            return

        results = self.ingester.ingest_all()
        self.vector_store.add_results(results)
        self._initialized = True
        logger.info(f"Initialized EvaluationRAG with {len(results)} results")

    def query(self, question: str, top_k: int = 5) -> str:
        """Query the evaluation knowledge base."""
        self.initialize()

        results = self.vector_store.search(question, top_k)

        if not results:
            return "No relevant evaluation data found for your query."

        # Format results as context
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            context_parts.append(f"""
Result {i}:
- App: {metadata.get('app_name')} v{metadata.get('app_version')}
- Task: {metadata.get('task_type')}
- Batch: {metadata.get('batch_id')}
- Metrics: {json.dumps(metadata.get('metrics', {}), indent=2)}
- Documents evaluated: {len(metadata.get('per_document_scores', {}))}
""")

        return "\n".join(context_parts)

    def get_task_summary(self, task_type: str) -> str:
        """Get a summary of all evaluations for a task type."""
        self.initialize()

        results = self.vector_store.get_by_task(task_type)

        if not results:
            return f"No evaluation data found for task: {task_type}"

        summary_parts = [f"## Evaluation Summary for {task_type}\n"]
        summary_parts.append(f"Total evaluations: {len(results)}\n")

        for result in results:
            metadata = result['metadata']
            metrics = metadata.get('metrics', {})
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            summary_parts.append(
                f"- **{metadata.get('app_name')}** (v{metadata.get('app_version')}): {metrics_str}"
            )

        # Add best performer
        best = self.vector_store.get_best_for_task(task_type)
        if best:
            summary_parts.append(f"\n**Best performer:** {best['metadata'].get('app_name')}")

        return "\n".join(summary_parts)

    def get_app_performance(self, app_name: str) -> str:
        """Get performance data for a specific app across all tasks."""
        self.initialize()

        results = self.vector_store.get_by_app(app_name)

        if not results:
            return f"No evaluation data found for app: {app_name}"

        summary_parts = [f"## Performance Summary for {app_name}\n"]

        for result in results:
            metadata = result['metadata']
            metrics = metadata.get('metrics', {})
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            summary_parts.append(
                f"- **{metadata.get('task_type')}** ({metadata.get('batch_id')}): {metrics_str}"
            )

            if metadata.get('limitations'):
                summary_parts.append(f"  - Limitations: {metadata['limitations'][0][:100]}...")

        return "\n".join(summary_parts)

    def recommend_pipeline(self, task_description: str) -> str:
        """Recommend apps based on task description and historical performance."""
        self.initialize()

        # Search for relevant evaluations
        results = self.vector_store.search(task_description, top_k=10)

        if not results:
            return "Unable to find relevant evaluation data for recommendations."

        # Group by task type and find best performers
        recommendations = {}
        for result in results:
            task_type = result['metadata'].get('task_type')
            if task_type not in recommendations:
                best = self.vector_store.get_best_for_task(task_type)
                if best:
                    recommendations[task_type] = best['metadata']

        # Format recommendations
        rec_parts = ["## Recommended Apps Based on Evaluation Data\n"]

        for task_type, metadata in recommendations.items():
            metrics = metadata.get('metrics', {})
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            rec_parts.append(
                f"**{task_type}:** {metadata.get('app_name')} v{metadata.get('app_version')}\n"
                f"  - Performance: {metrics_str}\n"
                f"  - Batch: {metadata.get('batch_id')}"
            )

        return "\n".join(rec_parts)


def create_evaluation_rag_tools(rag: EvaluationRAG):
    """Create LangChain tools from EvaluationRAG instance."""
    from langchain_core.tools import tool

    @tool("query_evaluation_data")
    def query_evaluation_data(question: str) -> str:
        """Query the CLAMS evaluation knowledge base for information about app performance.

        Use this to find out how different CLAMS apps have performed in evaluations,
        including metrics like precision, recall, F1, WER, CER, etc.

        Args:
            question: A natural language question about app performance or evaluations.

        Returns:
            Relevant evaluation results and metrics.
        """
        return rag.query(question)

    @tool("get_task_evaluation_summary")
    def get_task_evaluation_summary(task_type: str) -> str:
        """Get a summary of all evaluations for a specific task type.

        Task types include: AutomaticSpeechRecognition, TextRecognition,
        TimeFrameLabeling, TimePointLabeling, NamedEntityRecognition,
        NamedEntityGrounding, ForcedAlignment, KeyedInformationExtraction.

        Args:
            task_type: The evaluation task type to summarize.

        Returns:
            Summary of all app evaluations for that task, including best performer.
        """
        return rag.get_task_summary(task_type)

    @tool("get_app_performance_history")
    def get_app_performance_history(app_name: str) -> str:
        """Get the historical performance data for a specific CLAMS app.

        Use this to understand how an app has performed across different
        evaluation tasks and datasets.

        Args:
            app_name: The name of the CLAMS app (e.g., 'whisper-wrapper', 'parseqocr').

        Returns:
            Performance metrics across all evaluations for the app.
        """
        return rag.get_app_performance(app_name)

    @tool("recommend_apps_for_task")
    def recommend_apps_for_task(task_description: str) -> str:
        """Get app recommendations based on a task description and historical performance.

        Analyzes the task description and recommends the best-performing apps
        based on evaluation data.

        Args:
            task_description: Description of what you want to accomplish.

        Returns:
            Recommended apps with their historical performance metrics.
        """
        return rag.recommend_pipeline(task_description)

    return [
        query_evaluation_data,
        get_task_evaluation_summary,
        get_app_performance_history,
        recommend_apps_for_task
    ]


# Singleton instance
_evaluation_rag = None

def get_evaluation_rag() -> EvaluationRAG:
    """Get or create the EvaluationRAG singleton."""
    global _evaluation_rag
    if _evaluation_rag is None:
        _evaluation_rag = EvaluationRAG()
    return _evaluation_rag
