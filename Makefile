# ===============================
# Project Config
# ===============================

PROJECT_NAME=ai-platform
PYTHON=python3
VENV=.venv
APP=app.main:app

# Detect OS
ifeq ($(OS),Windows_NT)
	PYTHON=py -3.11
	VENV_BIN=$(VENV)\Scripts
	PIP=$(VENV_BIN)\pip
	UVICORN=$(VENV_BIN)\uvicorn
	RM=Remove-Item -Recurse -Force
else
	VENV_BIN=$(VENV)/bin
	PIP=$(VENV_BIN)/pip
	UVICORN=$(VENV_BIN)/uvicorn
	RM=rm -rf
endif

# ===============================
# Virtual Environment
# ===============================

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install -r requirements.txt

freeze:
	$(PIP) freeze > requirements.txt

# ===============================
# Run Server
# ===============================

run:
	$(UVICORN) $(APP) --reload

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
	$(RM) $(VENV)