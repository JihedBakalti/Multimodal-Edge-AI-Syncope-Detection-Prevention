# Platform + Orchestrator Deployment Notes

## Canonical Identity
- Use Firebase/ESP `user_id` as the canonical patient key across both services.
- In Django, `PatientProfile.firebase_user_id` stores this value and must match orchestrator payload `user_id`.

## Ingestion Contract (Orchestrator -> Platform)
Required payload fields:
- `user_id`
- `timestamp` (ISO8601 UTC)
- `idempotency_key`
- `state`
- `session_id`
- `source`

Common clinical fields:
- `heart_rate`
- `blood_oxygen`
- `anomaly_value`
- `dl_risk_score`
- `actions`
- `model_name`
- `threshold`
- `metadata`

Headers:
- `X-Platform-Ingest-Token: <shared_secret>`

## Orchestrator Identification (MVP)
- Treat orchestrator as a service identity (machine client), not a human login.
- Keep static token auth for MVP/testing.
- For production, replace static token with HMAC or service JWT without changing payload schema.

## Environment Variables

### Platform backend
- `PLATFORM_INGEST_TOKEN`
- `PLATFORM_INGEST_REPLAY_WINDOW_SEC` (default `300`)
- `PLATFORM_DB_ENGINE` (`sqlite` or `postgres`)
- `PLATFORM_DB_NAME`
- `PLATFORM_DB_USER`
- `PLATFORM_DB_PASSWORD`
- `PLATFORM_DB_HOST`
- `PLATFORM_DB_PORT`
- `PLATFORM_DB_SSLMODE`

### Orchestrator (both API entry points)
- `PLATFORM_INGEST_URL` (e.g. `https://platform.example.com/api/ingestion/orchestrator-event/`)
- `PLATFORM_INGEST_TOKEN`
- `PLATFORM_INGEST_RETRIES` (default `3`)
- `PLATFORM_INGEST_TIMEOUT_SEC` (default `4`)
- `PLATFORM_INGEST_BACKOFF_SEC` (default `1.0`)
- `ORCH_DEFAULT_USER_ID` (fallback if request has no `user_id`)

## SQLite vs Production
- SQLite is acceptable for local testing only.
- For split-server production, run PostgreSQL and set `PLATFORM_DB_ENGINE=postgres`.
