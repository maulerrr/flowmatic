# Flowmatic Sensor Simulator

Standalone demo service that emits synthetic smart-city sensor payloads over **HTTP polling** and **WebSocket streaming**. Flowmatic pipelines connect to this service in `SIMULATED` source mode or as an `EXTERNAL` endpoint during local development and Docker Compose demos.

## Quick start (local)

```powershell
cd services/sensor-simulator
copy .env.example .env
bun install
bun run dev
```

Service URL: `http://localhost:8091`

## Environment

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `PORT` | `8091` | HTTP + WebSocket port |
| `HOST` | `0.0.0.0` | Bind address |
| `DEFAULT_LOCATION` | `Astana` | Location field in payloads |
| `API_KEY` | _(empty)_ | Optional shared secret (`X-API-Key` header) |
| `CORS_ORIGINS` | `*` | Comma-separated CORS origins |

## API

### Health

`GET /health`

### Presets (for UI + backend auto-config)

`GET /api/v1/presets`

Returns ready-to-use HTTP and WebSocket URLs per sensor kind (`iot`, `video`, `power`, `network`, `weather`, `parking`).

### HTTP polling

`GET /api/v1/poll?sensorKind=iot&limit=5`

Response:

```json
{
  "events": [{ "...": "..." }],
  "sensorKind": "iot",
  "count": 5,
  "generatedAt": "2026-05-21T12:00:00.000Z"
}
```

Flowmatic reads the `events` array via optional `payloadPath=events` in source connection config.

### Single emit (test)

`POST /api/v1/emit?sensorKind=weather`

### WebSocket stream

`ws://localhost:8091/ws?sensorKind=network&intervalMs=1000`

Sends one JSON payload immediately, then repeats on the configured interval. Send `ping` to receive `{ "type": "pong" }`.

## Docker

```powershell
docker build -t flowmatic-sensor-simulator .
docker run --rm -p 8091:8091 --env-file .env flowmatic-sensor-simulator
```

Or use the root `docker-compose.yml` stack (`sensor-simulator` service).

## Wiring in Flowmatic

Set in backend `.env`:

```env
SENSOR_SIMULATOR_URL=http://localhost:8091
```

In Docker Compose the backend uses `http://sensor-simulator:8091`.

When creating a pipeline source with mode `SIMULATED`, Flowmatic resolves endpoints against this service automatically.
