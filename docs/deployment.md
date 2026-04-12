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
4. Set `API_BEARER_TOKEN` before exposing the API outside local development.
5. Make sure `TRADING_MODE` and `ALPACA_PAPER` match.
6. Set `MODEL_HMAC_SECRET` so model integrity checks do not rely on the development default.
7. If you want DSI forecasts active, set all of `DSI_BASE_URL`, `DSI_EMAIL`, and `DSI_PASSWORD`, and use `https`.
8. Run:

```bash
docker compose up -d --build
```

This starts:

- `api` on port 8000
- `worker` for scheduled trading cycles and retraining

## Shared state requirements

If you deploy the API and worker as separate services, do not rely on
per-service local files for shared dashboard state.

Use:

- Supabase for trades, predictions, equity, and runtime state
- a persistent disk for model artifacts on the worker

The worker now publishes runtime state such as heartbeat, latest cycle,
portfolio heat, and adaptive model-performance snapshots so the API can
report meaningful health information even when it runs on a separate host.

## Hardening for production

- Put the API behind Nginx or Caddy with HTTPS.
- Restrict dashboard access with auth.
- Set `API_BEARER_TOKEN` and pass it to the dashboard as `?token=...` or via browser local storage.
- Keep `TRADING_MODE=paper`, `ENABLE_LIVE_TRADING=false`, and `ALPACA_PAPER=true` for first cloud rollout.
- Treat `/health` as the deployment health probe. It now returns `503` when the worker heartbeat is stale or core dependencies are degraded.
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
