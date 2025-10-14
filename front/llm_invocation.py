import streamlit as st
from datetime import datetime, timedelta
from hashlib import sha256
from difflib import SequenceMatcher
from sqlalchemy.orm import Session, joinedload
from front.helpers import (
    ensure_user_exists,
    simulate_model_inference,
    invoke_openai,
    invoke_deepseek,
    calculate_cost,
    get_user_budget
)
from app.db.models import PromptCache, ProjectBudget, ModelUsageLog

def hash_prompt(prompt: str, model_name: str, max_tokens: int) -> str:
    """
    Generates a unique hash key for a prompt combined with model name and max_tokens.
    Normalizes prompt by stripping, lowering, and collapsing spaces.
    """
    import re
    normalized_prompt = re.sub(r'\s+', ' ', prompt.strip().lower())
    return sha256(f"{normalized_prompt}|{model_name}|{max_tokens}".encode()).hexdigest()

def get_cached_response(session: Session, key: str):
    """
    Retrieves an exact cached response using the prompt hash key.
    """
    cached = session.query(PromptCache).filter_by(prompt_key=key).first()
    if cached:
        st.write(f"Debug: Exact cache hit for key {key}")
        return {
            "response": cached.response,
            "input_tokens": cached.input_tokens,
            "output_tokens": cached.output_tokens,
            "timestamp": cached.timestamp
        }
    st.write(f"Debug: No exact cache match for key {key}")
    return None

def get_similar_cached_prompt(session: Session, prompt: str, model_name: str, max_tokens: int,
                              similarity_threshold=0.8, ttl_days=30):
    """
    Finds a similar prompt from the cache based on string similarity and matching model and tokens.
    Only scans recent cache entries for given model_name and max_tokens.
    """
    now = datetime.now()

    recent_caches = session.query(PromptCache).filter(
        PromptCache.model_name == model_name,
        PromptCache.max_tokens == max_tokens,
        PromptCache.timestamp != None,
        PromptCache.timestamp > now - timedelta(days=ttl_days)
    ).all()

    best_match = None
    best_ratio = 0.0
    normalized_prompt = prompt.strip().lower()

    for item in recent_caches:
        if not item.raw_prompt:
            continue
        similarity = SequenceMatcher(None, normalized_prompt, item.raw_prompt.strip().lower()).ratio()
        if similarity > best_ratio:
            best_ratio = similarity
            best_match = item

    if best_match and best_ratio >= similarity_threshold:
        st.write(f"Debug: Similar cache hit with similarity {best_ratio:.2%}")
        return {
            "response": best_match.response,
            "input_tokens": best_match.input_tokens,
            "output_tokens": best_match.output_tokens,
            "similarity": best_ratio,
            "match_type": "similar"
        }
    st.write(f"Debug: No similar cache match above threshold {similarity_threshold}")
    return None

def cache_response(session: Session, key: str, raw_prompt: str, model_name: str, max_tokens: int,
                   response: str, input_tokens: int, output_tokens: int):
    """
    Caches or updates the result of a prompt inference.
    """
    now = datetime.now()
    existing = session.query(PromptCache).filter_by(prompt_key=key).first()
    if existing:
        existing.response = response
        existing.input_tokens = input_tokens
        existing.output_tokens = output_tokens
        existing.timestamp = now
        existing.raw_prompt = raw_prompt
        existing.model_name = model_name
        existing.max_tokens = max_tokens
    else:
        session.add(PromptCache(
            prompt_key=key,
            raw_prompt=raw_prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=now
        ))
    session.commit()

def display_llm_invocation(session: Session, user_email: str, project_budget: ProjectBudget,
                           usage_history, models, weeks, use_demo_mode: bool):
    """
    Displays the Streamlit LLM invocation interface.
    """
    st.title("LLM Invocation")

    if not models:
        st.error("No models found in the database.")
        return

    if user_email:
        ensure_user_exists(session, user_email)
        project_budget = get_user_budget(session, user_email)

    selected_model = st.selectbox("Model", models, format_func=lambda m: m.name)
    prompt = st.text_area("Prompt", height=150)
    max_tokens = st.number_input("Max Tokens", min_value=1, value=1000)
    invocation_mode = st.selectbox("Invocation Mode", ["Simulation", "OpenAI", "DeepSeek"])
    api_key = st.text_input(f"{invocation_mode} API Key", type="password") if invocation_mode != "Simulation" else ""
    bypass_cache = st.checkbox("Bypass Cache")

    st.markdown(f"**Remaining Budget**: {project_budget.currency} {project_budget.amount:.2f}")

    if st.button("Invoke LLM"):
        try:
            selected_model_fresh = session.get(
                type(selected_model),
                selected_model.id,
                options=[joinedload(type(selected_model).pricings)]
            )

            if not selected_model_fresh:
                st.error("Selected model not found in DB.")
                return

            pricing = max(selected_model_fresh.pricings, key=lambda p: p.updated_at) if selected_model_fresh.pricings else None
            if not pricing:
                st.error("No pricing info for selected model.")
                return

            if not (pricing.input_cost or pricing.cached_input or pricing.output_cost):
                st.warning("Pricing data is zero or missing. Using defaults: $5/million input, $15/million output.")

            key = hash_prompt(prompt, selected_model.name, max_tokens)

            response = ""
            input_tokens = output_tokens = cached_input = 0
            use_cache = False
            cache_hit_type = ""

            if not bypass_cache:
                cached_result = get_cached_response(session, key)
                if cached_result:
                    response = cached_result["response"]
                    input_tokens = cached_result["input_tokens"]
                    output_tokens = cached_result["output_tokens"]
                    cached_input = input_tokens
                    use_cache = True
                    cache_hit_type = "Exact match"
                else:
                    similar_result = get_similar_cached_prompt(session, prompt, selected_model.name, max_tokens)
                    if similar_result:
                        response = similar_result["response"]
                        input_tokens = similar_result["input_tokens"]
                        output_tokens = similar_result["output_tokens"]
                        cached_input = input_tokens
                        use_cache = True
                        cache_hit_type = f"Similar match ({similar_result['similarity']:.2%})"

            if not use_cache:
                if invocation_mode == "Simulation":
                    response, input_tokens, output_tokens = simulate_model_inference(prompt, max_tokens)
                elif invocation_mode == "OpenAI":
                    if not api_key:
                        raise Exception("OpenAI API key required.")
                    response, input_tokens, output_tokens = invoke_openai(prompt, max_tokens, api_key, selected_model.name)
                elif invocation_mode == "DeepSeek":
                    if not api_key:
                        raise Exception("DeepSeek API key required.")
                    response, input_tokens, output_tokens = invoke_deepseek(prompt, max_tokens, api_key, selected_model.name)
                else:
                    raise Exception("Unsupported invocation mode.")

                cache_response(session, key, prompt, selected_model.name, max_tokens, response, input_tokens, output_tokens)

            cost = calculate_cost(pricing, input_tokens, output_tokens, use_cache=use_cache)

            if cost > project_budget.amount:
                st.error("Not enough budget to invoke this model.")
                return

            usage_log = ModelUsageLog(
                user_mail=user_email,
                model_name=selected_model.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input=cached_input,
                cost=cost,
                timestamp=datetime.now()
            )
            session.add(usage_log)

            if not use_demo_mode:
                project_budget.amount -= cost
                if project_budget.id is not None:
                    session.query(ProjectBudget).filter_by(id=project_budget.id).update({"amount": project_budget.amount})
                    session.commit()
                else:
                    st.warning("Budget not saved to database as it's a default budget.")
            else:
                session.rollback()

            st.success("Response generated successfully.")
            st.markdown("### LLM Response")
            st.write(response)
            st.markdown("---")
            st.markdown(
                f"**Input Tokens:** {input_tokens}\n\n"
                f"**Output Tokens:** {output_tokens}\n\n"
                f"**Cached Input:** {cached_input}\n\n"
                f"**Cost:** {project_budget.currency} {cost:.6f}\n\n"
                f"**Updated Budget:** {project_budget.currency} {project_budget.amount:.2f}"
            )
            if use_cache:
                st.info(f"Cache Hit: {cache_hit_type}")

        except Exception as e:
            session.rollback()
            st.error(f"Error during LLM invocation: {e}")