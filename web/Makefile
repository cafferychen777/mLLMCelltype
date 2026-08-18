# mLLMCelltype Web Makefile

.PHONY: help check deploy lock lock-upgrade test clean run logs status

PYTHON := python3

help:
	@echo "Available commands:"
	@echo "  make check    - Run pre-deploy code quality checks"
	@echo "  make deploy   - Deploy to VPS (push to main)"
	@echo "  make lock     - Refresh lock files while preserving compatible pins"
	@echo "  make lock-upgrade - Upgrade all dependencies and refresh lock files"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean temporary files"
	@echo "  make run      - Start local dev server"
	@echo "  make logs     - View VPS container logs"
	@echo "  make status   - Check service health"

check:
	@echo "Running code quality checks..."
	@./scripts/pre_deploy_check.sh

deploy: check
	@echo "Pushing to main to trigger GitHub Actions deploy..."
	git push origin main

lock:
	@./scripts/compile_requirements.sh

lock-upgrade:
	@./scripts/compile_requirements.sh --upgrade-all

test:
	@echo "Running tests..."
	@pytest -q
	@$(PYTHON) scripts/check_frontend_methods.py
	@node --check static/js/app.js

clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -f .DS_Store 2>/dev/null || true
	@rm -f scripts/*.log 2>/dev/null || true
	@echo "Done"

run:
	@echo "Starting dev server..."
	@$(PYTHON) app.py

logs:
	@echo "Fetching VPS logs..."
	@curl -sf https://www.mllmcelltype.com/health && echo " (healthy)" || echo " (unreachable)"

status:
	@echo "Checking service health..."
	@curl -sf https://www.mllmcelltype.com/health | python3 -m json.tool 2>/dev/null || echo "Service unreachable"
