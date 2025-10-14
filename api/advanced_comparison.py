import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional
from pydantic import BaseModel

from app.db.db_setup import get_db
from app.db.models import (
    Model, Provider, Pricing, Benchmark, BenchmarkResult,
    ModelTask, Task
)

router = APIRouter(prefix="/advancedaggregation")
logger = logging.getLogger(__name__)


# === Pydantic Models ===

class CostPerformanceMetric(BaseModel):
    """Cost performance metrics for a model based on benchmark results."""
    model_id: int
    model_name: str
    provider_name: str
    benchmark_name: str
    score: float
    cost_per_point: float
    cost_efficiency_rank: int


class OpenSourceModelOut(BaseModel):
    """Output model for open-source models with no pricing."""
    model_id: int
    model_name: str
    provider_name: str
    parameters: Optional[int]
    context_window: Optional[int]
    tasks: List[str]


class CostOptimizedModel(BaseModel):
    """Model with cost optimization metrics."""
    model_id: int
    model_name: str
    provider: str
    input_cost: float
    output_cost: float
    total_cost: float
    avg_benchmark_score: Optional[float]
    cost_efficiency: Optional[float]


class CostOptimizationResponse(BaseModel):
    """Response model for cost optimization endpoint."""
    task_name: str
    optimized_models: List[CostOptimizedModel]


# === Endpoints ===

@router.get(
    "/best-value-models",
    response_model=List[CostPerformanceMetric],
    summary="Get top cost-efficient models by benchmark",
    description="""
Retrieve AI models ranked by **cost-efficiency** for a given benchmark.

This endpoint helps identify models that offer the **best trade-off between cost and performance**.
It calculates a metric called `cost_per_point` = average cost ÷ benchmark score.

The results are:
- **Sorted by lowest cost_per_point**
- **Paginated**
- **Globally ranked**

**Note**: Only models with a benchmark score ≥ `min_score` are considered.
"""
)
def get_best_value_models(
    benchmark_name: str = Query(..., description="Name of the benchmark (e.g. 'MMLU')", min_length=1, max_length=100),
    page: int = Query(1, ge=1, le=1000, description="Page number (starts at 1, max 1000)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of results per page (1-100)"),
    min_score: float = Query(1.0, ge=0.1, le=100.0, description="Minimum benchmark score (0.1-100) to consider"),
    db: Session = Depends(get_db)
):
    try:
        benchmark_exists = db.query(Benchmark.id).filter(
            func.lower(Benchmark.name) == benchmark_name.lower().strip()
        ).first()

        if not benchmark_exists:
            available_benchmarks = [b[0] for b in db.query(Benchmark.name).limit(10).all()]
            raise HTTPException(
                status_code=422,
                detail=f"Benchmark '{benchmark_name}' not found. Available: {available_benchmarks}"
            )

        latest_pricing_subq = (
            db.query(
                Pricing.model_id,
                Pricing.input_cost,
                Pricing.output_cost,
                func.row_number().over(
                    partition_by=Pricing.model_id,
                    order_by=desc(Pricing.created_at)
                ).label('rn')
            ).subquery()
        )

        base_query = (
            db.query(
                Model.id,
                Model.name,
                Provider.name.label('provider_name'),
                Benchmark.name.label('benchmark_name'),
                BenchmarkResult.score,
                latest_pricing_subq.c.input_cost,
                latest_pricing_subq.c.output_cost,
                (((latest_pricing_subq.c.input_cost + latest_pricing_subq.c.output_cost) / 2.0) / BenchmarkResult.score).label('cost_per_point')
            )
            .join(Provider, Model.provider_id == Provider.id)
            .join(BenchmarkResult, BenchmarkResult.model_id == Model.id)
            .join(Benchmark, BenchmarkResult.benchmark_id == Benchmark.id)
            .join(
                latest_pricing_subq,
                and_(
                    latest_pricing_subq.c.model_id == Model.id,
                    latest_pricing_subq.c.rn == 1
                )
            )
            .filter(
                func.lower(Benchmark.name) == benchmark_name.lower().strip(),
                BenchmarkResult.score.isnot(None),
                BenchmarkResult.score >= min_score,
                latest_pricing_subq.c.input_cost.isnot(None),
                latest_pricing_subq.c.output_cost.isnot(None),
                latest_pricing_subq.c.input_cost >= 0,
                latest_pricing_subq.c.output_cost >= 0
            )
            .order_by('cost_per_point', Model.name)
        )

        total_count = base_query.count()
        if total_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No models found for benchmark '{benchmark_name}' with score >= {min_score} and valid pricing"
            )

        offset = (page - 1) * page_size
        results = base_query.offset(offset).limit(page_size).all()

        metrics = []
        for i, row in enumerate(results):
            global_rank = offset + i + 1
            metrics.append(CostPerformanceMetric(
                model_id=row.id,
                model_name=row.name,
                provider_name=row.provider_name,
                benchmark_name=row.benchmark_name,
                score=float(row.score),
                cost_per_point=round(float(row.cost_per_point), 6),
                cost_efficiency_rank=global_rank
            ))

        logger.info(
            f"Retrieved {len(metrics)}/{total_count} models for '{benchmark_name}' "
            f"(page {page}/{(total_count + page_size - 1) // page_size}, min_score={min_score})"
        )

        return metrics

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_best_value_models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in get_best_value_models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/open-source-models", response_model=List[OpenSourceModelOut])
def get_open_source_models(
    task_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Récupère la liste des modèles open-source (modèles sans tarification associée ou gratuits).

    Un modèle est considéré comme open-source si aucun tarif n’est défini ou si les coûts 
    d’entrée et de sortie sont nuls.

    Paramètres :
    - task_name (str, optionnel) : nom de la tâche pour filtrer les modèles supportant cette tâche.
      Si aucun nom n’est fourni, tous les modèles open-source sont retournés.
    - db (Session) : session de base de données injectée par FastAPI.

    Retour :
    - Liste de `OpenSourceModelOut` : chaque élément contient les informations du modèle, 
      y compris ses tâches associées.

    Comportement :
    - Si `task_name` est fourni mais n’existe pas dans la base, une liste vide est retournée.
    - Pour chaque modèle retourné, la liste des tâches supportées est également incluse.

    Exemple d’appel :
    ```
    GET /open-source-models?task_name=text-generation
    ```

    Réponse (exemple simplifié) :
    [
        {
            "model_id": 1,
            "model_name": "GPT-2",
            "provider_name": "OpenAI",
            "parameters": 1500000000,
            "context_window": 1024,
            "tasks": ["text-generation", "summarization"]
        },
        ...
    ]
    """

    query = db.query(
        Model.id.label("model_id"),
        Model.name.label("model_name"),
        Provider.name.label("provider_name"),
        Model.parameters,
        Model.context_window
    ).join(
        Provider, Model.provider_id == Provider.id
    ).outerjoin(
        Pricing, Model.id == Pricing.model_id
    ).filter(
        (Pricing.id == None) | ((Pricing.input_cost == 0) & (Pricing.output_cost == 0))
    )

    if task_name:
        task = db.query(Task).filter(Task.task_name == task_name).first()
        if not task:
            return []
        query = query.join(ModelTask, Model.id == ModelTask.model_id).filter(
            ModelTask.task_id == task.id
        )

    models = query.all()
    model_ids = [m.model_id for m in models]
    if not model_ids:
        return []

    task_map = {mid: [] for mid in model_ids}
    all_tasks = (
        db.query(ModelTask.model_id, Task.task_name)
        .join(Task, Task.id == ModelTask.task_id)
        .filter(ModelTask.model_id.in_(model_ids))
        .all()
    )
    for mid, tname in all_tasks:
        task_map[mid].append(tname)

    return [
        OpenSourceModelOut(
            model_id=m.model_id,
            model_name=m.model_name,
            provider_name=m.provider_name,
            parameters=m.parameters,
            context_window=m.context_window,
            tasks=task_map.get(m.model_id, [])
        )
        for m in models
    ]

