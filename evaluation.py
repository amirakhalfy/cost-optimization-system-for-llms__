import os
import logging
from langsmith import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Évalue automatiquement des modèles Hugging Face via LangSmith"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        if not self.api_key:
            raise ValueError("LANGSMITH_API_KEY must be set")
        self.client = Client(api_key=self.api_key)

    def generate_prompts(self, model_info: dict):
        """Génère des prompts simples à partir des infos du modèle"""
        name = model_info.get("model_name", "unknown_model")
        prompts = [
            {
                "input": f"Using {name}: Translate 'Hello world!' to French.",
                "expected": "Bonjour le monde !"
            },
            {
                "input": f"Using {name}: Summarize: 'LangSmith simplifies model evaluation.'",
                "expected": "LangSmith facilite l'évaluation des modèles."
            }
        ]
        return prompts

    def evaluate_model(self, model_info: dict):
        """Évalue un modèle Hugging Face avec LangSmith"""
        metrics_result = {}
        model_name = model_info.get("model_name", "unknown_model")
        prompts = self.generate_prompts(model_info)

        for test_case in prompts:
            try:
                result = self.client.track_run(
                    model=model_name,
                    input=test_case["input"],
                    expected_output=test_case["expected"]
                )
                # On peut mapper les résultats dans un dictionnaire clair
                metrics_result[test_case["input"]] = {
                    "competence": result.get("competence_score", 0),
                    "accuracy": result.get("accuracy_score", 0),
                    "robustness": result.get("robustness_score", 0),
                    "relevance": result.get("relevance_score", 0)
                }
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation du modèle {model_name}: {e}")
                metrics_result[test_case["input"]] = {
                    "competence": 0,
                    "accuracy": 0,
                    "robustness": 0,
                    "relevance": 0
                }

        # Moyenne des metrics
        averaged_metrics = {
            "competence": sum([v["competence"] for v in metrics_result.values()]) / len(metrics_result),
            "accuracy": sum([v["accuracy"] for v in metrics_result.values()]) / len(metrics_result),
            "robustness": sum([v["robustness"] for v in metrics_result.values()]) / len(metrics_result),
            "relevance": sum([v["relevance"] for v in metrics_result.values()]) / len(metrics_result),
        }

        return {"model_info": model_info, "metrics": averaged_metrics}
