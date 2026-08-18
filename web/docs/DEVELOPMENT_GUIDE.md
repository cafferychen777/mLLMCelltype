# Development Guide

## Template syntax

Flask uses `{[{` and `}]}` as Jinja delimiters because Vue uses `{{` and `}}`.
Use `{[{ url_for(...) }]}` for server-rendered values in every template. See
`TEMPLATE_SYNTAX_GUIDE.md` for examples.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
cp .env.example .env
make check
python app.py
```

## Required checks

`make check` is the deployment gate. It runs:

- Python compilation
- Ruff over application, utility, script, and test code
- pytest
- Vue template contract validation
- JavaScript syntax checks for static and inline scripts
- a tracked-file credential scan
- a check that `uv.lock` is not tracked
- a provenance check tying both dependency locks to their source requirements
- a check that Docker excludes `.env` and `.git` from its build context
- production shell and Docker Compose contract checks

Run the same command before pushing. The GitHub Actions deployment workflow runs
it again before connecting to the production server.

## Design constraints

- Keep code and comments in English.
- Treat the task state machine as the only place that changes task states.
- Hold `TASKS_LOCK` while mutating shared task dictionaries, but never during
  database or provider network calls.
- Never persist API keys.
- Read provider defaults from `mllmcelltype.config`; do not duplicate them in
  web-specific code.
- Use `utils.time_utils` for UTC timestamps and
  `utils.serialization.to_json_compatible` at JSON boundaries.
