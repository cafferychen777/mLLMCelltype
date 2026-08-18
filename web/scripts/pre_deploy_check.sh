#!/usr/bin/env bash
set -euo pipefail

echo "Running Python syntax checks..."
python3 -m compileall -q app.py config storage utils scripts tests

echo "Checking deployment shell syntax..."
bash -n scripts/*.sh infra/ansible/files/deploy-mllmcelltype \
    infra/ansible/files/check-mllmcelltype
if command -v shellcheck >/dev/null 2>&1; then
    shellcheck scripts/*.sh infra/ansible/files/deploy-mllmcelltype \
        infra/ansible/files/check-mllmcelltype
fi

echo "Running Ruff..."
ruff check app.py config storage utils scripts tests
ruff format --check app.py config storage utils scripts tests

echo "Running tests..."
pytest -q

echo "Checking frontend contracts..."
python3 scripts/check_frontend_methods.py

if command -v node >/dev/null 2>&1; then
    echo "Checking JavaScript syntax..."
    node --check static/js/app.js
    python3 - <<'PY'
import re
import subprocess
from pathlib import Path

for path in Path("templates").glob("*.html"):
    content = path.read_text(encoding="utf-8")
    scripts = re.findall(
        r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>",
        content,
        re.DOTALL,
    )
    for index, (attributes, script) in enumerate(scripts, 1):
        if "src=" in attributes or "application/ld+json" in attributes:
            continue
        result = subprocess.run(
            ["node", "--check"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise SystemExit(
                f"{path} inline script {index} failed syntax check:\n{result.stderr}"
            )
PY
fi

echo "Scanning tracked files for credential-shaped values..."
if git grep -I -n -E '(sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,}|(AKIA|ASIA)[A-Z0-9]{16}|TURSO_AUTH_TOKEN=(eyJ|[A-Za-z0-9_-]{40})|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY)' -- .; then
    echo "Credential-shaped value found in tracked files." >&2
    exit 1
fi

if [[ -e uv.lock ]] || git ls-files --error-unmatch uv.lock >/dev/null 2>&1; then
    echo "uv.lock must not exist or be committed." >&2
    exit 1
fi

echo "Checking dependency lock provenance..."
./scripts/compile_requirements.sh --check

if ! grep -Fxq '.env' .dockerignore || ! grep -Fxq '.git' .dockerignore; then
    echo ".dockerignore must exclude .env and .git from the image context." >&2
    exit 1
fi

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "Validating the production Compose contract..."
    compose_config=$(APP_IMAGE=mllmcelltype-web:check APP_ENV_FILE=/dev/null \
        docker compose -f docker-compose.production.yml config)
    if ! grep -Fq 'host_ip: 127.0.0.1' <<<"$compose_config" || \
        ! grep -Fq 'published: "8080"' <<<"$compose_config"; then
        echo "The application must bind only to loopback port 8080." >&2
        exit 1
    fi
    if grep -Eq '^  caddy:|^[[:space:]]+build:' <<<"$compose_config"; then
        echo "Compose must not build images or own public ingress." >&2
        exit 1
    fi
fi

echo "All pre-deploy checks passed."
