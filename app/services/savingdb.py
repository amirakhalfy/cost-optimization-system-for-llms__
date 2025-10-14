from datetime import datetime
from sqlalchemy import func
from app.db.models import Provider, Model, Pricing, Benchmark, Task, ModelTask, BenchmarkResult, ModelHistory
from app.db.db_setup import SessionLocal
import re

def safe_float(value, default=0.0):
    """Safely convert a value to float, handling currency symbols and returning default if conversion fails."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = re.sub(r'[^\d.-]', '', value)
        try:
            return float(value)
        except ValueError:
            return default
    return default

def safe_int(value, default=None):
    """Safely convert a value to int, returning default if conversion fails."""
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def normalize_name(name):
    """Normalize model or task names by removing spaces, hyphens, and underscores, and converting to lowercase."""
    if not name:
        return ""
    return re.sub(r'[-\s_]', '', name.lower())

def safe_lower(value):
    """Safely convert a value to lowercase, handling None values."""
    if value is None:
        return ""
    return str(value).lower()

def safe_currency(value, default="USD"):
    """Safely get currency value, ensuring it's never None or empty."""
    if not value or value is None:
        return default
    currency_str = str(value).strip().upper()
    return currency_str if currency_str else default

def normalize_tasks_list(tasks_list):
    """Normalize and clean tasks list."""
    if not tasks_list:
        return set()
    
    if isinstance(tasks_list, str):
        tasks_list = [t.strip() for t in re.split(r'[,;|]', tasks_list)]
    elif not isinstance(tasks_list, list):
        return set()
    
    return {t.strip() for t in tasks_list if t and isinstance(t, str) and t.strip()}

def save_to_database(all_models):
    """
    Save extracted AI model data to the database, avoiding duplicates and updating only when pricing or benchmarks change.
    For existing models, update pricing and benchmarks only if they differ, saving old pricing to ModelHistory.
    Tasks are updated by adding new ones or removing obsolete ones. New models are added as needed.

    Args:
        all_models (list): List of dictionaries containing model data.

    Returns:
        str: A success message if data is saved/updated, None if an error occurs.
    """
    session = SessionLocal()
    try:
        updated_models = 0
        new_models = 0

        for model_data in all_models:
            # --- Provider ---
            provider_name = model_data.get('provider') or "Unknown Provider"
            provider = session.query(Provider).filter(func.lower(Provider.name) == safe_lower(provider_name)).first()
            if not provider:
                provider = Provider(
                    name=provider_name,
                    website=model_data.get('website', ""),
                    created_at=datetime.now()
                )
                session.add(provider)
                session.flush()
            provider_id = provider.id

            # --- Model ---
            model_name = model_data.get('model_name')
            if not model_name:
                continue
            
            # Fixed: Use normalize_name for both sides of comparison
            norm_model_name = normalize_name(model_name)
            existing_model = session.query(Model).filter(
                Model.provider_id == provider_id
            ).all()
            
            # Find existing model by comparing normalized names
            found_model = None
            for model in existing_model:
                if normalize_name(model.name) == norm_model_name:
                    found_model = model
                    break

            if found_model:
                model_id = found_model.id
                has_changes = False

                # --- Pricing ---
                pricing_data = model_data.get('pricing')
                latest_pricing = session.query(Pricing).filter_by(model_id=model_id).order_by(Pricing.created_at.desc()).first()
                new_pricing = None
                pricing_changed = False
                if pricing_data and isinstance(pricing_data, dict):
                    unit = pricing_data.get('unit', 'token')
                    unit_str = str(unit).lower() if unit else 'token'
                    token_unit = 1000000 if 'token' in unit_str else 1000 if 'request' in unit_str else 1000000
                    
                    # Extract pricing values with proper defaults and currency handling
                    new_pricing = {
                        'input_cost': safe_float(pricing_data.get('input_cost'), 0.0),
                        'output_cost': safe_float(pricing_data.get('output_cost'), 0.0),
                        'cached_input': safe_float(pricing_data.get('cached_input'), 0.0),
                        'training_cost': safe_float(pricing_data.get('training_cost'), 0.0),
                        'token_unit': token_unit,
                        'currency': safe_currency(pricing_data.get('currency'), "USD")
                    }

                    if latest_pricing:
                        latest_values = {
                            'input_cost': latest_pricing.input_cost or 0.0,
                            'output_cost': latest_pricing.output_cost or 0.0,
                            'cached_input': latest_pricing.cached_input or 0.0,
                            'training_cost': latest_pricing.training_cost or 0.0,
                            'token_unit': latest_pricing.token_unit or 1000000,
                            'currency': latest_pricing.currency or "USD"
                        }
                        pricing_changed = any(latest_values[key] != new_pricing[key] for key in new_pricing)
                        if pricing_changed:
                            session.add(ModelHistory(
                                model_id=model_id,
                                input_cost=latest_values['input_cost'],
                                output_cost=latest_values['output_cost'],
                                training_cost=latest_values['training_cost'],
                                currency=latest_values['currency'],
                                created_at=datetime.now()
                            ))
                            session.add(Pricing(
                                model_id=model_id,
                                **new_pricing,
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            ))
                            has_changes = True
                    else:
                        session.add(Pricing(
                            model_id=model_id,
                            **new_pricing,
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        ))
                        has_changes = True
                else:
                    if not latest_pricing:
                        session.add(Pricing(
                            model_id=model_id,
                            input_cost=0.0,
                            output_cost=0.0,
                            cached_input=0.0,
                            training_cost=0.0,
                            token_unit=1000000,
                            currency="USD",
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        ))
                        has_changes = True

                # --- Update Model Info (Fixed: Update existing model with new data) ---
                model_updated = False
                if found_model.description != model_data.get('description'):
                    found_model.description = model_data.get('description')
                    model_updated = True
                if found_model.license != model_data.get('license'):
                    found_model.license = model_data.get('license')
                    model_updated = True
                if found_model.context_window != safe_int(model_data.get('context_window')):
                    found_model.context_window = safe_int(model_data.get('context_window'))
                    model_updated = True
                if found_model.max_tokens != safe_int(model_data.get('max_tokens')):
                    found_model.max_tokens = safe_int(model_data.get('max_tokens'))
                    model_updated = True
                if found_model.parameters != safe_int(model_data.get('parameters')):
                    found_model.parameters = safe_int(model_data.get('parameters'))
                    model_updated = True
                
                if model_updated:
                    has_changes = True

                # --- Benchmarks ---
                benchmarks_data = model_data.get('benchmarks')
                benchmarks_changed = False
                if benchmarks_data and isinstance(benchmarks_data, dict):
                    for name, score in benchmarks_data.items():
                        if not name or score is None:
                            continue
                        score_float = safe_float(score)
                        benchmark = session.query(Benchmark).filter(func.lower(Benchmark.name) == safe_lower(name)).first()
                        if not benchmark:
                            benchmark = Benchmark(
                                name=name,
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            session.add(benchmark)
                            session.flush()
                        result = session.query(BenchmarkResult).filter_by(
                            model_id=model_id,
                            benchmark_id=benchmark.id
                        ).first()
                        if result:
                            if result.score != score_float:
                                result.score = score_float
                                result.evaluation_date = datetime.now()
                                benchmarks_changed = True
                        else:
                            session.add(BenchmarkResult(
                                model_id=model_id,
                                benchmark_id=benchmark.id,
                                score=score_float,
                                evaluation_date=datetime.now(),
                                created_at=datetime.now()
                            ))
                            benchmarks_changed = True

                # --- Optimized Tasks Processing ---
                tasks_list = model_data.get('tasks')
                tasks_changed = False
                
                # Normalize incoming tasks
                new_tasks = normalize_tasks_list(tasks_list)
                
                # Only process if there are tasks to check
                if new_tasks or tasks_list is not None:  # Check if tasks_list was provided (even if empty)
                    # Get current tasks efficiently
                    current_task_results = session.query(Task.task_name).join(ModelTask).filter(ModelTask.model_id == model_id).all()
                    current_task_names = {normalize_name(result.task_name) for result in current_task_results}
                    new_task_names = {normalize_name(task) for task in new_tasks}
                    
                    # Early exit if tasks are identical
                    if current_task_names == new_task_names:
                        # Tasks are identical, no changes needed
                        pass
                    else:
                        # Tasks have changed, process the differences
                        tasks_to_add = new_task_names - current_task_names
                        tasks_to_remove = current_task_names - new_task_names
                        
                        # Add new tasks
                        if tasks_to_add:
                            for new_task_name in new_tasks:
                                norm_task_name = normalize_name(new_task_name)
                                if norm_task_name in tasks_to_add:
                                    task = session.query(Task).filter(func.lower(Task.task_name) == norm_task_name).first()
                                    if not task:
                                        task = Task(
                                            task_name=new_task_name,
                                            task_category='unknown',
                                            created_at=datetime.now(),
                                            updated_at=datetime.now()
                                        )
                                        session.add(task)
                                        session.flush()
                                    session.add(ModelTask(model_id=model_id, task_id=task.id))
                                    tasks_changed = True

                        # Remove obsolete tasks
                        if tasks_to_remove:
                            # Get tasks that need to be removed with their IDs
                            tasks_to_remove_objs = session.query(Task).join(ModelTask).filter(
                                ModelTask.model_id == model_id,
                                func.lower(Task.task_name).in_([name for name in tasks_to_remove])
                            ).all()
                            
                            for task in tasks_to_remove_objs:
                                mt = session.query(ModelTask).filter_by(model_id=model_id, task_id=task.id).first()
                                if mt:
                                    session.delete(mt)
                                    tasks_changed = True

                # Update model only if pricing, benchmarks, or tasks changed
                if pricing_changed or benchmarks_changed or tasks_changed or has_changes:
                    found_model.updated_at = datetime.now()
                    updated_models += 1
                    session.flush()

            else:
                # --- New Model ---
                new_model = Model(
                    name=model_name,  # Keep original name with spaces
                    provider_id=provider_id,
                    license=model_data.get('license'),
                    description=model_data.get('description'),
                    context_window=safe_int(model_data.get('context_window')),
                    max_tokens=safe_int(model_data.get('max_tokens')),
                    parameters=safe_int(model_data.get('parameters')),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                session.add(new_model)
                session.flush()
                model_id = new_model.id
                new_models += 1

                # --- Pricing for New Model ---
                pricing_data = model_data.get('pricing')
                if pricing_data and isinstance(pricing_data, dict):
                    unit = pricing_data.get('unit', 'token')
                    unit_str = str(unit).lower() if unit else 'token'
                    token_unit = 1000000 if 'token' in unit_str else 1000 if 'request' in unit_str else 1000000
                    
                    # Extract pricing values with proper defaults and currency handling
                    new_pricing = {
                        'input_cost': safe_float(pricing_data.get('input_cost'), 0.0),
                        'output_cost': safe_float(pricing_data.get('output_cost'), 0.0),
                        'cached_input': safe_float(pricing_data.get('cached_input'), 0.0),
                        'training_cost': safe_float(pricing_data.get('training_cost'), 0.0),
                        'token_unit': token_unit,
                        'currency': safe_currency(pricing_data.get('currency'), "USD")
                    }
                    session.add(Pricing(
                        model_id=model_id,
                        **new_pricing,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    ))
                else:
                    session.add(Pricing(
                        model_id=model_id,
                        input_cost=0.0,
                        output_cost=0.0,
                        cached_input=0.0,
                        training_cost=0.0,
                        token_unit=1000000,
                        currency="USD",
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    ))

                # --- Benchmarks for New Model ---
                benchmarks_data = model_data.get('benchmarks')
                if benchmarks_data and isinstance(benchmarks_data, dict):
                    for name, score in benchmarks_data.items():
                        if not name or score is None:
                            continue
                        score_float = safe_float(score)
                        benchmark = session.query(Benchmark).filter(func.lower(Benchmark.name) == safe_lower(name)).first()
                        if not benchmark:
                            benchmark = Benchmark(
                                name=name,
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            session.add(benchmark)
                            session.flush()
                        session.add(BenchmarkResult(
                            model_id=model_id,
                            benchmark_id=benchmark.id,
                            score=score_float,
                            evaluation_date=datetime.now(),
                            created_at=datetime.now()
                        ))

                # --- Tasks for New Model ---
                new_tasks = normalize_tasks_list(model_data.get('tasks'))
                if new_tasks:
                    for task_name in new_tasks:
                        norm_task_name = normalize_name(task_name)
                        task = session.query(Task).filter(func.lower(Task.task_name) == norm_task_name).first()
                        if not task:
                            task = Task(
                                task_name=task_name,
                                task_category='unknown',
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            session.add(task)
                            session.flush()
                        session.add(ModelTask(model_id=model_id, task_id=task.id))

        session.commit()
        msg = f"Successfully processed {len(all_models)} models: {new_models} new models added, {updated_models} models updated."
        print(msg)
        return msg

    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}")
        return None

    finally:
        session.close()