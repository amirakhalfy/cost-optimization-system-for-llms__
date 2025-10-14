from httpcore import request
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from pydantic import BaseModel , EmailStr
from typing import Optional , List
from app.db.db_setup import SessionLocal
from app.db.models import Provider, Model, Pricing, Benchmark, BenchmarkResult, Task, ModelTask,User ,ProjectBudget , UserProjectBudget
from app.db.db_setup import get_db
from datetime import datetime

session = SessionLocal()
router = APIRouter(prefix="/sql")

class ProviderBase(BaseModel):
    """Base model for Provider"""
    name: str
    description: Optional[str] = None

class ModelBase(BaseModel):
    """Base model for Model"""
    name: str
    provider_id: int
    description: Optional[str] = None

class PricingBase(BaseModel):
    """Base model for Pricing"""
    model_id: int
    input_cost: float
    output_cost: float
    currency: str = "USD"

class BenchmarkBase(BaseModel):
    """Base model for Benchmark"""
    name: str
    description: Optional[str] = None

class BenchmarkResultBase(BaseModel):
    """Base model for BenchmarkResult"""
    model_id: int
    benchmark_id: int
    score: float
    timestamp: Optional[str] = None

class TaskBase(BaseModel):
    """Base model for Task"""
    name: str
    description: Optional[str] = None

class ModelTaskBase(BaseModel):
    """Base model for ModelTask"""
    model_id: int
    task_id: int


class UserBase(BaseModel):
    """Base model for User"""
    user_mail: EmailStr
    role: str

class UserCreate(UserBase):
    """Model for creating a new user"""
    pass

class UserResponse(UserBase):
    """Response model for User"""
    created_at: datetime
    class Config:
        from_attributes = True

class ProjectBudgetBase(BaseModel):
    """Base model for ProjectBudget"""
    name: str
    amount: float
    description: Optional[str] = None
    currency: str
    period: str
    alert_threshold: float

class ProjectBudgetCreate(ProjectBudgetBase):
    """Model for creating a new project budget"""
    user_mail: str

class ProjectBudgetResponse(ProjectBudgetBase):
    """Response model for ProjectBudget"""  
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    users: List[UserResponse] = []
@router.get("/providers")
def get_providers(db: Session = Depends(get_db)):
    """Retrieve all providers
   example response:
    [
        {
            "id": 1,
            "name": "Provider A",
            "description": "Description of Provider A"
        },
        {
            "id": 2,
            "name": "Provider B",
            "description": "Description of Provider B"
        }
    ]
    """
    providers = db.query(Provider).all()
    if not providers:
        raise HTTPException(status_code=404, detail="Aucun fournisseur trouvé")
    return providers

@router.post("/providers", status_code=status.HTTP_201_CREATED)
def create_provider(provider: ProviderBase, db: Session = Depends(get_db)):
    """Create a new provider
    example request body:
    {
        "name": "Provider A",
        "description": "Description of Provider A"
    }
    """
    new_provider = Provider(**provider.dict())
    db.add(new_provider)
    db.commit()
    db.refresh(new_provider)
    return new_provider

@router.put("/providers/{provider_id}")
def update_provider(provider_id: int, provider: ProviderBase, db: Session = Depends(get_db)):
    """"Update an existing provider"""
    db_provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not db_provider:
        raise HTTPException(status_code=404, detail="Fournisseur non trouvé")
    
    for key, value in provider.dict().items():
        setattr(db_provider, key, value)
    
    db.commit()
    db.refresh(db_provider)
    return db_provider

@router.delete("/providers/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_provider(provider_id: int, db: Session = Depends(get_db)):
    """Delete a provider"""
    db_provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not db_provider:
        raise HTTPException(status_code=404, detail="Fournisseur non trouvé")
    
    db.delete(db_provider)
    db.commit()
    return None

@router.get("/models")
def get_models(db: Session = Depends(get_db)):
    """Retrieve all models
    example response:
    [
        {
            "id": 1,
            "name": "Model A",
            "description": "Description of Model A"
        },
        {
            "id": 2,
            "name": "Model B",
            "description": "Description of Model B"
        }
    ]
    """
    models = db.query(Model).all()
    if not models:
        raise HTTPException(status_code=404, detail="Aucun modèle trouvé")
    return models

@router.post("/models", status_code=status.HTTP_201_CREATED)
def create_model(model: ModelBase, db: Session = Depends(get_db)):
    """Create a new model"""
    provider = db.query(Provider).filter(Provider.id == model.provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Fournisseur non trouvé")
        
    new_model = Model(**model.dict())
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model

@router.put("/models/{model_id}")
def update_model(model_id: int, model: ModelBase, db: Session = Depends(get_db)):
    """Update an existing model"""
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    provider = db.query(Provider).filter(Provider.id == model.provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Fournisseur non trouvé")
    
    for key, value in model.dict().items():
        setattr(db_model, key, value)
    
    db.commit()
    db.refresh(db_model)
    return db_model

@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete a model"""
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    db.delete(db_model)
    db.commit()
    return None

@router.get("/pricing")
def get_pricing(db: Session = Depends(get_db)):
    """Retrieve all pricing information"""
    pricings = db.query(Pricing).all()
    if not pricings:
        raise HTTPException(status_code=404, detail="Aucun tarif trouvé")
    return pricings

@router.post("/pricing", status_code=status.HTTP_201_CREATED)
def create_pricing(pricing: PricingBase, db: Session = Depends(get_db)):
    """Create a new pricing entry"""
    model = db.query(Model).filter(Model.id == pricing.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
        
    new_pricing = Pricing(**pricing.dict())
    db.add(new_pricing)
    db.commit()
    db.refresh(new_pricing)
    return new_pricing

@router.put("/pricing/{pricing_id}")
def update_pricing(pricing_id: int, pricing: PricingBase, db: Session = Depends(get_db)):
    """Update an existing pricing entry"""
    db_pricing = db.query(Pricing).filter(Pricing.id == pricing_id).first()
    if not db_pricing:
        raise HTTPException(status_code=404, detail="Tarif non trouvé")
    
    model = db.query(Model).filter(Model.id == pricing.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    for key, value in pricing.dict().items():
        setattr(db_pricing, key, value)
    
    db.commit()
    db.refresh(db_pricing)
    return db_pricing

@router.delete("/pricing/{pricing_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_pricing(pricing_id: int, db: Session = Depends(get_db)):
    """Delete a pricing entry"""
    db_pricing = db.query(Pricing).filter(Pricing.id == pricing_id).first()
    if not db_pricing:
        raise HTTPException(status_code=404, detail="Tarif non trouvé")
    
    db.delete(db_pricing)
    db.commit()
    return None

@router.get("/benchmarks")
def get_benchmarks(db: Session = Depends(get_db)):
    """Retrieve all benchmarks"""
    benchmarks = db.query(Benchmark).all()
    if not benchmarks:
        raise HTTPException(status_code=404, detail="Aucun benchmark trouvé")
    return benchmarks

@router.post("/benchmarks", status_code=status.HTTP_201_CREATED)
def create_benchmark(benchmark: BenchmarkBase, db: Session = Depends(get_db)):
    """Create a new benchmark"""
    new_benchmark = Benchmark(**benchmark.dict())
    db.add(new_benchmark)
    db.commit()
    db.refresh(new_benchmark)
    return new_benchmark

@router.put("/benchmarks/{benchmark_id}")
def update_benchmark(benchmark_id: int, benchmark: BenchmarkBase, db: Session = Depends(get_db)):
    """Update an existing benchmark"""
    db_benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not db_benchmark:
        raise HTTPException(status_code=404, detail="Benchmark non trouvé")
    
    for key, value in benchmark.dict().items():
        setattr(db_benchmark, key, value)
    
    db.commit()
    db.refresh(db_benchmark)
    return db_benchmark

@router.delete("/benchmarks/{benchmark_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_benchmark(benchmark_id: int, db: Session = Depends(get_db)):
    """Delete a benchmark"""
    db_benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not db_benchmark:
        raise HTTPException(status_code=404, detail="Benchmark non trouvé")
    
    db.delete(db_benchmark)
    db.commit()
    return None

@router.get("/benchmarkresults")
def get_benchmark_results(db: Session = Depends(get_db)):
    """Retrieve all benchmark results"""
    results = db.query(BenchmarkResult).all()
    if not results:
        raise HTTPException(status_code=404, detail="Aucun résultat de benchmark trouvé")
    return results

@router.post("/benchmarkresults", status_code=status.HTTP_201_CREATED)
def create_benchmark_result(result: BenchmarkResultBase, db: Session = Depends(get_db)):
    """Create a new benchmark result"""
    model = db.query(Model).filter(Model.id == result.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    benchmark = db.query(Benchmark).filter(Benchmark.id == result.benchmark_id).first()
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark non trouvé")
        
    new_result = BenchmarkResult(**result.dict())
    db.add(new_result)
    db.commit()
    db.refresh(new_result)
    return new_result

@router.put("/benchmarkresults/{result_id}")
def update_benchmark_result(result_id: int, result: BenchmarkResultBase, db: Session = Depends(get_db)):
    """Update an existing benchmark result"""
    db_result = db.query(BenchmarkResult).filter(BenchmarkResult.id == result_id).first()
    if not db_result:
        raise HTTPException(status_code=404, detail="Résultat de benchmark non trouvé")
    
    model = db.query(Model).filter(Model.id == result.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    benchmark = db.query(Benchmark).filter(Benchmark.id == result.benchmark_id).first()
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark non trouvé")
    
    for key, value in result.dict().items():
        setattr(db_result, key, value)
    
    db.commit()
    db.refresh(db_result)
    return db_result

@router.delete("/benchmarkresults/{result_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_benchmark_result(result_id: int, db: Session = Depends(get_db)):
    """Delete a benchmark result"""
    db_result = db.query(BenchmarkResult).filter(BenchmarkResult.id == result_id).first()
    if not db_result:
        raise HTTPException(status_code=404, detail="Résultat de benchmark non trouvé")
    
    db.delete(db_result)
    db.commit()
    return None

@router.get("/tasks")
def get_tasks(db: Session = Depends(get_db)):
    """Retrieve all tasks"""
    tasks = db.query(Task).all()
    if not tasks:
        raise HTTPException(status_code=404, detail="Aucune tâche trouvée")
    return tasks

@router.post("/tasks", status_code=status.HTTP_201_CREATED)
def create_task(task: TaskBase, db: Session = Depends(get_db)):
    """Create a new task"""
    new_task = Task(**task.dict())
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return new_task

@router.put("/tasks/{task_id}")
def update_task(task_id: int, task: TaskBase, db: Session = Depends(get_db)):
    """Update an existing task"""
    db_task = db.query(Task).filter(Task.id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    for key, value in task.dict().items():
        setattr(db_task, key, value)
    
    db.commit()
    db.refresh(db_task)
    return db_task

@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    """Delete a task"""
    db_task = db.query(Task).filter(Task.id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    db.delete(db_task)
    db.commit()
    return None

@router.get("/modeltasks")
def get_model_tasks(db: Session = Depends(get_db)):
    """Retrieve all model-task links"""
    model_tasks = db.query(ModelTask).all()
    if not model_tasks:
        raise HTTPException(status_code=404, detail="Aucun lien modèle-tâche trouvé")
    return model_tasks

@router.post("/modeltasks", status_code=status.HTTP_201_CREATED)
def create_model_task(model_task: ModelTaskBase, db: Session = Depends(get_db)):
    """Create a new model-task link"""
    model = db.query(Model).filter(Model.id == model_task.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    task = db.query(Task).filter(Task.id == model_task.task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
        
    new_model_task = ModelTask(**model_task.dict())
    db.add(new_model_task)
    db.commit()
    db.refresh(new_model_task)
    return new_model_task

@router.put("/modeltasks/{model_task_id}")
def update_model_task(model_task_id: int, model_task: ModelTaskBase, db: Session = Depends(get_db)):
    """Update an existing model-task link"""
    db_model_task = db.query(ModelTask).filter(ModelTask.id == model_task_id).first()
    if not db_model_task:
        raise HTTPException(status_code=404, detail="Lien modèle-tâche non trouvé")
    
    model = db.query(Model).filter(Model.id == model_task.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    task = db.query(Task).filter(Task.id == model_task.task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    for key, value in model_task.dict().items():
        setattr(db_model_task, key, value)
    
    db.commit()
    db.refresh(db_model_task)
    return db_model_task

@router.delete("/modeltasks/{model_task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model_task(model_task_id: int, db: Session = Depends(get_db)):
    """Delete a model-task link"""
    db_model_task = db.query(ModelTask).filter(ModelTask.id == model_task_id).first()
    if not db_model_task:
        raise HTTPException(status_code=404, detail="Lien modèle-tâche non trouvé")
    
    db.delete(db_model_task)
    db.commit()
    return None

@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    db_user = db.query(User).filter(User.user_mail == user.user_mail).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Utilisateur existe déjà")
    
    new_user = User(**user.dict())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.delete("/users/{user_mail}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_mail: str, db: Session = Depends(get_db)):
    """Delete a user by email"""
    db_user = db.query(User).filter(User.user_mail == user_mail).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    db.delete(db_user)
    db.commit()
    return None

@router.get("/users", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    """Retrieve all users"""
    users = db.query(User).all()
    if not users:
        raise HTTPException(status_code=404, detail="Aucun utilisateur trouvé")
    return users

@router.put("/users/{user_mail}", response_model=UserResponse)
def update_user(user_mail: str, user: UserBase, db: Session = Depends(get_db)):
    """Update an existing user by email"""
    db_user = db.query(User).filter(User.user_mail == user_mail).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    for key, value in user.dict().items():
        setattr(db_user, key, value)
    
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/budgets", response_model=ProjectBudgetResponse, status_code=status.HTTP_201_CREATED)
def create_project_budget(budget: ProjectBudgetCreate, db: Session = Depends(get_db)):
    """Create a new project budget"""
    user = db.query(User).filter(User.user_mail == budget.user_mail).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    new_budget = ProjectBudget(
        name=budget.name,
        description=budget.description,
        amount=budget.amount,
        currency=budget.currency,
        period=budget.period,
        alert_threshold=budget.alert_threshold,
        is_active=True,
    )
    
    db.add(new_budget)
    db.commit()
    db.refresh(new_budget)

    user_project_budget = UserProjectBudget(
        user_mail=budget.user_mail,
        project_budget_id=new_budget.id
    )
    db.add(user_project_budget)
    db.commit()

    return new_budget