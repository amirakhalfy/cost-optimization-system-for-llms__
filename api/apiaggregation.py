import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Dict, List, Optional
from pydantic import BaseModel
from app.db.db_setup import get_db
from app.db.models import Model, Provider, Pricing, Benchmark, Task, ModelTask, BenchmarkResult
from sqlalchemy.orm import joinedload

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/aggregation")

class ModelInfo(BaseModel):
    """Model information including ID, name, and optionally tasks covered."""
    id: int
    name: str
    tasks_covered: Optional[List[str]] = None

class ModelRecommendation(BaseModel):
    """Model recommendation for a specific task, including best overall, budget, and performance models."""
    task_name: str
    best_overall_model: dict
    best_budget_model: dict
    best_performance_model: dict

class ModelComparison(BaseModel):
    """Comparison of models based on pricing and tasks."""
    model_id: int
    model_name: str
    provider_name: str
    input_cost: float
    output_cost: float
    cached_input: float
    token_unit: str
    currency: str
    tasks: List[str]
    context_window: Optional[int]
    parameters: Optional[int]
class ProviderInfoResponse(BaseModel):
    """Response model for provider information."""
    provider_name: str
    model_count: int
    avg_input_cost: float
    avg_output_cost: float
    benchmark_performance: Dict[str, float]
    tasks_covered: List[str]

@router.get("/available-models", response_model=List[ModelInfo])
def get_available_models(
    include_tasks: bool = Query(False, description="Include the list of tasks supported by each model"),
    db: Session = Depends(get_db)
):
    """
    Retrieve a list of all available models with their IDs and optionally their supported tasks.

    This endpoint is critical as it enables users to discover valid model names and IDs for use in other API endpoints,
    such as model comparison, ensuring accurate and efficient interactions with the system.

    Parameters:
    ----------
    include_tasks : bool, optional
        If true, includes the list of tasks supported by each model. Defaults to False.

    Returns:
    -------
    List[ModelInfo]
        A list of dictionaries containing model IDs, names, and optionally supported tasks.

    Raises:
    ------
    HTTPException (404)
        If no models are found in the database.
    """
    logger.info("Récupération des modèles disponibles")
    if include_tasks:
        models = db.query(Model).options(joinedload(Model.tasks)).all()
    else:
        models = db.query(Model).all()

    if not models:
        logger.warning("Aucun modèle trouvé dans la base de données")
        raise HTTPException(status_code=404, detail="No models found in the database")

    model_infos = []
    for model in models:
        model_info = {"id": model.id, "name": model.name}
        if include_tasks:
            model_info["tasks_covered"] = [task.task_name for task in model.tasks]
        model_infos.append(ModelInfo(**model_info))

    logger.info(f"Nombre de modèles récupérés : {len(model_infos)}")
    return model_infos

@router.get("/compare-models", response_model=List[ModelComparison])
def compare_models(
    model_names: List[str] = Query(..., description="List of model names to compare. Must match available model names from /available-models"),
    task_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Compare AI models by pricing, supported tasks, and technical specifications.

    Args:
        model_names (List[str]): List of model names to compare.
        task_name (Optional[str]): Filter tasks by this name (optional).
        db (Session): Database session.

    Returns:
        List[ModelComparison]: List of models with pricing, tasks, and technical specifications.

    Raises:
        HTTPException 404: If no models found or missing pricing for all.
    """
    logger.info(f"Comparaison des modèles : {model_names}")
    selected_models = db.query(Model).options(joinedload(Model.provider)) \
        .filter(Model.name.in_(model_names)).all()

    found_names = {model.name for model in selected_models}
    missing_models = [name for name in model_names if name not in found_names]

    if not selected_models:
        logger.warning(f"Modèles non trouvés : {', '.join(missing_models)}")
        raise HTTPException(status_code=404, detail=f"Models not found: {', '.join(missing_models)}")

    model_ids = [m.id for m in selected_models]
    latest_pricing_by_model_id = {
        p.model_id: p for p in db.query(Pricing)
        .filter(Pricing.model_id.in_(model_ids))
        .distinct(Pricing.model_id)
        .order_by(Pricing.model_id, desc(Pricing.created_at))
        .all()
    }

    task_query = db.query(ModelTask.model_id, Task.task_name).join(Task).filter(ModelTask.model_id.in_(model_ids))
    if task_name:
        task_query = task_query.filter(func.lower(Task.task_name) == task_name.lower().strip())

    tasks_per_model = {}
    for model_id, task in task_query.all():
        tasks_per_model.setdefault(model_id, []).append(task)

    models_data = []
    missing_pricing = []

    for model in selected_models:
        pricing = latest_pricing_by_model_id.get(model.id)
        if not pricing:
            missing_pricing.append(model.name)
            continue

        tasks = tasks_per_model.get(model.id, [])
        provider_name = model.provider.name if model.provider else "Unknown"

        models_data.append(ModelComparison(
            model_id=model.id,
            model_name=model.name,
            provider_name=provider_name,
            input_cost=pricing.input_cost,
            output_cost=pricing.output_cost,
            cached_input=pricing.cached_input,
            token_unit=pricing.token_unit,
            currency=pricing.currency,
            tasks=tasks,
            context_window=model.context_window,
            parameters=model.parameters
        ))

    if missing_models or missing_pricing:
        error_detail = []
        if missing_models:
            error_detail.append(f"Invalid or missing models: {', '.join(missing_models)}")
        if missing_pricing:
            error_detail.append(f"Models without pricing data: {', '.join(missing_pricing)}")
        if not models_data:
            logger.warning("Aucun modèle valide pour la comparaison")
            raise HTTPException(status_code=404, detail="; ".join(error_detail))
        logger.warning("Partial success: " + "; ".join(error_detail))

    logger.info(f"Comparaison terminée pour {len(models_data)} modèles")
    return models_data

@router.get("/provider-comparison")
def compare_providers(db: Session = Depends(get_db)):
    """
    Provides a consolidated view of AI model providers with key metrics for risk management 
    and cost optimization.

    This API enables:
    - Evaluation of provider diversity and robustness (number of models available).
    - Analysis of average usage costs (input and output costs per model).
    - Comparison of average performance across different benchmarks, aiding in quality-to-price assessment.
    - Identification of functional capabilities (tasks covered by the provider's models).

    Usage in a risk management and cost optimization system:
    -------------------------------------------------------
    - Monitoring supplier risk (dependency on a single provider, functional coverage).
    - Controlling expenses by tracking average model usage costs.
    - Optimized selection of providers and models based on performance and business needs.
    - Support for automated decision-making in budget allocation and strategic planning.

    Returns:
    --------
    List[dict]: A list of objects containing for each provider:
        - provider_id (int): Unique provider identifier
        - provider_name (str): Provider name
        - model_count (int): Total number of models offered
        - avg_input_cost (float): Average cost per input token
        - avg_output_cost (float): Average cost per output token
        - benchmark_performance (dict): Average performance by benchmark (name -> score)
        - tasks_covered (List[str]): List of functional tasks covered by the provider

    Raises:
    -------
    HTTPException (404): If no providers are found in the database.
    """
    logger.info("Retrieving providers from the database")
    providers = db.query(Provider).all()
    if not providers:
        logger.warning("No providers found in the database")
        raise HTTPException(status_code=404, detail="No providers found in the database")

    model_counts_subq = (
        db.query(
            Model.provider_id.label("provider_id"),
            func.count(Model.id).label("model_count")
        )
        .group_by(Model.provider_id)
        .subquery()
    )

    pricing_subq = (
        db.query(
            Model.provider_id.label("provider_id"),
            func.avg(Pricing.input_cost).label("avg_input_cost"),
            func.avg(Pricing.output_cost).label("avg_output_cost"),
        )
        .join(Pricing, Pricing.model_id == Model.id)
        .group_by(Model.provider_id)
        .subquery()
    )

    benchmark_subq = (
        db.query(
            Model.provider_id.label("provider_id"),
            Benchmark.name.label("benchmark_name"),
            func.avg(BenchmarkResult.score).label("avg_score"),
        )
        .join(BenchmarkResult, BenchmarkResult.model_id == Model.id)
        .join(Benchmark, Benchmark.id == BenchmarkResult.benchmark_id)
        .group_by(Model.provider_id, Benchmark.name)
        .subquery()
    )

    task_subq = (
        db.query(
            Model.provider_id.label("provider_id"),
            Task.task_name.label("task_name")
        )
        .join(ModelTask, ModelTask.model_id == Model.id)
        .join(Task, Task.id == ModelTask.task_id)
        .distinct()
        .subquery()
    )

    benchmarks_all = db.query(
        benchmark_subq.c.provider_id,
        benchmark_subq.c.benchmark_name,
        benchmark_subq.c.avg_score
    ).all()

    tasks_all = db.query(
        task_subq.c.provider_id,
        task_subq.c.task_name
    ).all()

    model_counts = dict(db.query(
        model_counts_subq.c.provider_id,
        model_counts_subq.c.model_count
    ).all())

    pricings = {
        row.provider_id: (row.avg_input_cost, row.avg_output_cost)
        for row in db.query(
            pricing_subq.c.provider_id,
            pricing_subq.c.avg_input_cost,
            pricing_subq.c.avg_output_cost
        ).all()
    }

    benchmarks_by_provider = {}
    for provider_id, benchmark_name, avg_score in benchmarks_all:
        benchmarks_by_provider.setdefault(provider_id, {})[benchmark_name] = avg_score

    tasks_by_provider = {}
    for provider_id, task_name in tasks_all:
        tasks_by_provider.setdefault(provider_id, set()).add(task_name)

    result = []
    for provider in providers:
        pid = provider.id
        if pid not in model_counts:
            continue

        result.append({
            "provider_id": pid,
            "provider_name": provider.name,
            "model_count": model_counts.get(pid, 0),
            "avg_input_cost": pricings.get(pid, (None, None))[0] or 0,
            "avg_output_cost": pricings.get(pid, (None, None))[1] or 0,
            "benchmark_performance": benchmarks_by_provider.get(pid, {}),
            "tasks_covered": list(tasks_by_provider.get(pid, []))
        })

    logger.info(f"Number of providers processed: {len(result)}")
    return result
@router.get("/provider-info/{provider_name}", response_model=ProviderInfoResponse)
def get_provider_info_by_name(provider_name: str, db: Session = Depends(get_db)):
    """
    Retrieve detailed information about a provider's models, pricing, benchmarks, and supported tasks.
    This endpoint provides a comprehensive overview of a specific provider, including:
    - The number of models offered by the provider.
    - Average input and output costs for the provider's models.
    - Benchmark performance metrics for the provider's models.
    - List of tasks covered by the provider's models.
    This information is crucial for evaluating the provider's capabilities and making informed decisions about model selection.
    Parameters:
    ----------
    provider_name : str
        The name of the provider to retrieve information for.
    db (Session): Database session dependency.
    Returns:
    -------
    ProviderInfoResponse: A response model containing provider information.
    Raises:
    ------
    HTTPException (404): If the provider is not found or has no models.
    HTTPException (500): If there is an internal server error while retrieving the provider information.

    """
    try:
        provider_id = db.query(Provider.id).filter(func.lower(Provider.name) == provider_name.lower()).scalar()
        if not provider_id:
            raise HTTPException(status_code=404, detail=f"Provider with name '{provider_name}' not found")

        model_count = db.query(func.count(Model.id)).filter(Model.provider_id == provider_id).scalar()
        if model_count == 0:
            raise HTTPException(status_code=404, detail=f"No models found for provider '{provider_name}'")

        avg_pricing = db.query(
            func.avg(Pricing.input_cost).label("avg_input_cost"),
            func.avg(Pricing.output_cost).label("avg_output_cost")
        ).join(Model, Pricing.model_id == Model.id).filter(Model.provider_id == provider_id).first()

        benchmark_performance = db.query(
            Benchmark.name,
            func.avg(BenchmarkResult.score).label("avg_score")
        ).join(BenchmarkResult, Benchmark.id == BenchmarkResult.benchmark_id)\
         .join(Model, BenchmarkResult.model_id == Model.id)\
         .filter(Model.provider_id == provider_id)\
         .group_by(Benchmark.name)\
         .all()

        tasks = db.query(Task.task_name)\
            .join(ModelTask, Task.id == ModelTask.task_id)\
            .join(Model, Model.id == ModelTask.model_id)\
            .filter(Model.provider_id == provider_id)\
            .distinct()\
            .all()

        return ProviderInfoResponse(
            provider_name=provider_name,
            model_count=model_count,
            avg_input_cost=float(avg_pricing.avg_input_cost or 0.0),
            avg_output_cost=float(avg_pricing.avg_output_cost or 0.0),
            benchmark_performance={name: float(score) for name, score in benchmark_performance},
            tasks_covered=[t[0] for t in tasks]
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations pour le fournisseur '{provider_name}': {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")
