# Thin wrapper over docker compose.
#
# It exists for one reason: the compose file lives in docker/ but the stack is
# built and configured from the repo root, and getting that combination right
# needs two flags that are easy to get wrong.
#
#   -f docker/docker-compose.yml   the file
#   --env-file .env                the ROOT .env, for ${...} substitution
#
# Why --env-file and not --project-directory: Compose takes its project directory
# from the compose file's location, and that is where it looks for .env. Pointing
# the project directory at the repo root fixes substitution but then resolves every
# relative path in the file against the root too — turning `env_file: ../.env` into
# a path outside the repo. --env-file changes only where variables come from, so
# `context: ..` and `env_file: ../.env` keep working.
#
# Always run these from the repo root.

COMPOSE := docker compose -f docker/docker-compose.yml --env-file .env

# The compose service name. Held in one place because it moved once already
# (app -> fastapi) and every target silently broke.
APP := fastapi

.PHONY: help up down build rebuild restart logs ps migrate shell db test load load-chat check monitoring targets

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up:  ## start the stack in the background
	$(COMPOSE) up -d

build:  ## build the app image
	$(COMPOSE) build $(APP)

rebuild:  ## rebuild and recreate the app container
	$(COMPOSE) build $(APP)
	$(COMPOSE) up -d --force-recreate $(APP)

restart:  ## recreate the app container without rebuilding (picks up .env changes)
	$(COMPOSE) up -d --force-recreate $(APP)

down:  ## stop the stack, keeping volumes
	$(COMPOSE) down

logs:  ## follow the app logs
	$(COMPOSE) logs -f $(APP)

ps:  ## show container status
	$(COMPOSE) ps

shell:  ## a shell inside the running app container
	$(COMPOSE) exec $(APP) sh

db:  ## a psql prompt on the database
	$(COMPOSE) exec pgvector psql -U $${POSTGRES_USERNAME:-postgres} -d $${POSTGRES_MAIN_DATABASE:-customer_support}

migrate:  ## apply migrations on the HOST (the container does this on boot too)
	uv run alembic upgrade head

test:  ## run the test suite
	uv run --extra dev pytest tests -q --no-cov

load:  ## read-only load test — free, no model calls
	uv run --extra dev locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 ReadOnlyUser

load-chat:  ## full-pipeline load test — SPENDS REAL MONEY and writes tickets
	uv run --extra dev locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 ConversationUser

check:  ## validate the compose file without starting anything
	$(COMPOSE) config --quiet && echo "compose OK"

monitoring:  ## where the dashboards are (all bound to localhost)
	@echo "  Grafana     http://localhost:3000   (see GF_SECURITY_ADMIN_* in .env)"
	@echo "  Prometheus  http://localhost:9090"
	@echo "  API         http://localhost:8000   direct, bypasses nginx"
	@echo "  nginx       http://localhost:80     the front door"

targets:  ## show which scrape targets Prometheus currently has up
	@curl -s localhost:9090/api/v1/targets \
		| python3 -c "import json,sys; [print(f\"  {t['labels']['job']:18} {t['health']}\") for t in json.load(sys.stdin)['data']['activeTargets']]"
