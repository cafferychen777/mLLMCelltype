# Deployment

Production is split into two independently managed layers:

- Host Caddy owns ports 80/443, certificates, redirects, and proxy logs.
- Docker Compose runs only the application image on `127.0.0.1:8080`.

The personal website is hosted by Cloudflare Pages and has no files, services,
certificates, or deployment credentials on this VPS.

## Release model

Changes under `web/` pushed to `main` start
`.github/workflows/web-deploy-vps.yml`. After tests pass, CI:

1. Builds `mllmcelltype-web:<full-commit-sha>` from the pinned Python base.
2. Stores the commit SHA in the image's OCI revision label.
3. Creates a checksummed bundle containing the image and Compose manifest.
4. Uploads the bundle with SSH host-key verification.
5. Invokes `/usr/local/sbin/deploy-mllmcelltype <sha>` through scoped sudo.

The root-owned deploy command validates the archive, checksum, revision label,
and Compose configuration. It waits for container and loopback health before
changing the atomic `current` symlink. On failure it restores the previous
release. Only the current and previous releases are retained.

The VPS does not hold a GitHub deploy key, repository checkout, registry token,
or permission to build releases.

## Host configuration

`infra/ansible/playbook.yml` is the idempotent source of truth for users, SSH,
Docker, Caddy, firewall rules, fail2ban, unattended upgrades, release paths,
health checks, and sudo policy. Run it from an operator workstation:

```bash
ansible-galaxy collection install -r infra/ansible/requirements.yml
ansible-playbook -i infra/ansible/inventory.yml infra/ansible/playbook.yml
```

The `ops` account uses an operator-only key and has administrative access. The
`deploy` account uses a separate CI key, is not in `sudo` or `docker`, and can
run only the release command. Required repository secrets are:

- `VPS_HOST`
- `VPS_PORT`
- `VPS_USERNAME` (always `deploy`)
- `VPS_SSH_KEY` (the dedicated CI private key)
- `VPS_HOST_FINGERPRINT`

Application secrets exist only in `/etc/mllmcelltype/app.env`, owned by root
with mode `0600`.

## Runtime model

Gunicorn uses one process with the `gthread` worker and eight request threads.
The single process is intentional: active task coordination is in memory, while
Turso provides durable snapshots. Annotation calls run in dedicated application
threads, so the HTTP request returns immediately.

Automatic worker recycling is disabled because it would terminate annotation
threads. A restart marks orphaned queued or processing tasks as failed instead
of presenting stale work as active. The container has a read-only root
filesystem, no Linux capabilities, a PID limit, and only a bounded temporary
filesystem.

## Operations

```bash
# Public and loopback health
curl --fail https://www.mllmcelltype.com/health
ssh ops@<VPS_IP> sudo curl --fail http://127.0.0.1:8080/health

# Application and ingress logs
ssh ops@<VPS_IP> sudo docker logs --tail 200 mllmcelltype-web-app-1
ssh ops@<VPS_IP> sudo journalctl -u caddy --since today

# Current and retained releases
ssh ops@<VPS_IP> sudo readlink -f /opt/mllmcelltype/current
ssh ops@<VPS_IP> sudo find /opt/mllmcelltype/releases -mindepth 1 -maxdepth 1 -type d
```

To roll back, choose the retained previous 40-character directory name and run:

```bash
ssh ops@<VPS_IP> sudo /usr/local/sbin/deploy-mllmcelltype <previous-sha>
```

GitHub Actions also checks production every 15 minutes. A host systemd timer
checks loopback health every two minutes and restarts only the application
container if it becomes unhealthy; Caddy remains available independently.

## Required environment

At minimum, `app.env` must define:

- `TURSO_DB_URL`
- `TURSO_AUTH_TOKEN`
- `FLASK_SECRET_KEY`

The admin dashboard is enabled only when both `ADMIN_USERNAME` and
`ADMIN_PASSWORD_HASH` are configured.
