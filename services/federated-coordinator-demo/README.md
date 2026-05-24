# Flowmatic Federated Coordinator Demo

Minimal HTTP/WebSocket coordinator used to demo federated learning connections from Flowmatic smart-city pipelines.

## Quick start

```powershell
cd services/federated-coordinator-demo
copy .env.example .env
bun install
bun run dev
```

Default URL: `http://localhost:8092`

## HTTP contract

`POST /api/v1/coordinator`

Accepts Flowmatic federated payloads (`type`, `projectId`, `nodeId`, `topic`, `data`, ...).

Supported types:

- `register` → `{ registrationId, globalModelVersion }`
- `pull_global_model` → `{ globalModelVersion, currentRoundId, rounds }`
- `round_started`, `round_update_submitted`, `round_aggregated` → acknowledgement

## WebSocket

`ws://localhost:8092/ws?nodeId=demo-node`

Auto-registers on connect. Send JSON messages using the same payload shape as HTTP.

## Flowmatic wiring

In the Connectors page federated panel:

- Protocol: `HTTP`
- Endpoint: `http://localhost:8092/api/v1/coordinator`

In Docker Compose use `http://federated-coordinator-demo:8092/api/v1/coordinator`.
