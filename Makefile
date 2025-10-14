start_dev:
	poetry run uvicorn --reload --host=0.0.0.0 --port=3206 app.main:app

alembic-init:
	poetry run alembic init alembic

run_db_only:
	docker-compose up -d db

alembic-revision-dev:
	poetry run alembic revision --autogenerate -m "$(msg)"

alembic-run-revision-dev:
	poetry run alembic upgrade head

alembic-revision-prod:
	docker exec -it otp_backend-app-1 alembic revision --autogenerate -m "$(msg)"

alembic-run-revision-prod:
	docker exec -it otp_backend-app-1 alembic upgrade head

restart:
	docker-compose down
	docker-compose up --build

streamlit:
	poetry run streamlit run front/main.py
fastapi:
	poetry run uvicorn app.main:app --reload
