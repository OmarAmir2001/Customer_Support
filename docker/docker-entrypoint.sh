#!/bin/sh
# Bring the schema up to date, then hand off to the real command.
#
# Migrations run here rather than in a separate one-shot service so that
# `docker compose up` is genuinely the only command needed. `alembic upgrade head`
# is idempotent, so restarts are free. If this ever runs as more than one replica,
# move it to its own init container — concurrent upgrades on one database race.
set -e

echo "waiting for the database to accept connections..."
python - <<'PY'
import os, time, socket, sys

host = os.environ.get("POSTGRES_HOST", "pgvector")
port = int(os.environ.get("POSTGRES_PORT", "5432"))

# compose's service_healthy already gates this, but a healthy Postgres can still
# refuse the first connection for a moment after it opens the port.
for attempt in range(30):
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"  {host}:{port} is accepting connections")
            sys.exit(0)
    except OSError:
        time.sleep(1)

print(f"  {host}:{port} never came up", file=sys.stderr)
sys.exit(1)
PY

echo "applying migrations..."
alembic upgrade head

echo "starting: $*"
exec "$@"
