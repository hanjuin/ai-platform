# ===============================
# Project Config
# ===============================

PROJECT_NAME=ai-platform
PYTHON=python
VENV=.venv
APP=app.main:app

# ===============================
# Virtual Environment
# ===============================

venv:
	py -3.11 -m venv $(VENV)

install:
	$(VENV)\Scripts\pip install -r requirements.txt

freeze:
	$(VENV)\Scripts\pip freeze > requirements.txt

# ===============================
# Run Server
# ===============================

run:
	$(VENV)\Scripts\uvicorn $(APP) --reload

# ===============================
# Docker
# ===============================

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-restart:
	docker compose down
	docker compose up -d

docker-logs:
	docker compose logs -f

docker-reset:
	docker compose down -v
	docker compose up -d

# ===============================
# Database
# ===============================

db-shell:
	docker exec -it ai_doc_db psql -U postgres -d aidoc

# ===============================
# Clean
# ===============================

clean:
	Remove-Item -Recurse -Force $(VENV)
