# Maintenance Scripts

Run these commands from the project root.

## Quality and dependency maintenance

```bash
./scripts/pre_deploy_check.sh
python scripts/check_frontend_methods.py
make lock
```

`make lock` preserves compatible versions already present in the lock files.
Use `./scripts/compile_requirements.sh --upgrade-package PACKAGE` for a targeted
dependency update, or `make lock-upgrade` for an explicit full dependency
refresh.

Production host configuration lives in `infra/ansible`; there is no
imperative setup script with unrestricted deployment privileges.

## Admin utilities

```bash
python scripts/admin/generate_admin_password.py
python scripts/admin/check_all_processing.py --base-url http://localhost:8080
python scripts/admin/check_task_status.py --task-id-prefix TASK_ID_PREFIX
```

Admin utilities read `ADMIN_USERNAME`, `ADMIN_PASSWORD`, and
`MLLM_WEB_BASE_URL` from the environment. Passwords are prompted for when
`ADMIN_PASSWORD` is unset.
