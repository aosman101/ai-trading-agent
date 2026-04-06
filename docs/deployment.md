# Deployment on a paid server

This project is designed to run well on a paid Ubuntu VM or container host.

## Recommended starting shape

For CPU-only inference and small nightly retraining:

- 4 vCPU
- 16 GB RAM
- 80+ GB SSD

If you want faster TFT / FinBERT work, add a separate GPU training worker later. Keep the execution worker on CPU.

## Docker deployment

1. Install Docker and Docker Compose.
2. Copy the project to the server.
3. Create a `.env` file from `.env.example`.
4. Run:

```bash
docker compose up -d --build
```

This starts:

- `api` on port 8000
- `worker` for scheduled trading cycles and retraining

## Hardening for production

- Put the API behind Nginx or Caddy with HTTPS.
- Restrict dashboard access with auth.
- Store secrets in a vault or cloud secret manager.
- Enable server monitoring and disk alerts.
- Add log shipping if you want long retention.
- Run model retraining in a separate service if training load grows.

## Scaling path

Phase 1:
- One VM
- API + worker together

Phase 2:
- One API instance
- One execution worker
- One training worker

Phase 3:
- Split broker execution, research training, and dashboard into separate services
- Add message queue and object storage for model artifacts
