# Production host configuration

This playbook owns host-level ingress, firewall rules, deploy privileges, and
the runtime health timer. Application releases remain the responsibility of the
GitHub Actions deployment workflow.

## Bootstrap

1. Keep the operator key in `~/.ssh/id_ed25519`, create a dedicated CI key at
   `~/.ssh/mllmcelltype_deploy`, and copy `inventory.example.yml` to the ignored
   `inventory.yml`. Set the VPS address there.
2. Install the required collection:

   ```bash
   ansible-galaxy collection install -r infra/ansible/requirements.yml
   ```

3. During the one-time migration from containerized Caddy, provision without
   activating host ingress:

   ```bash
   ansible-playbook infra/ansible/playbook.yml -e activate_caddy=false
   ```

4. After the application listens successfully on `127.0.0.1:8080`, stop the
   legacy Caddy container and apply the normal desired state. The playbook
   verifies public health before removing the legacy container, volumes,
   checkout, and outbound GitHub key:

   ```bash
   ansible-playbook infra/ansible/playbook.yml
   ```

The `ops` user is the human/Ansible administrator. The `deploy` user has a
different SSH key and can run only the validated release command; it is not a
member of `sudo` or `docker`.

Subsequent runs are idempotent. Secrets live only in
`/etc/mllmcelltype/app.env` on the host and are never copied into release
bundles or the repository.
