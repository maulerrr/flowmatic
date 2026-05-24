# Flowmatic Current State Handoff

Last updated: 2026-05-06

This file is a fresh-start handoff for another agent. It summarizes what is already implemented, what was validated, what is still missing, and what to check first before making more changes.

## 1. Repo and working assumptions

- Repo root: `C:\Users\BG\Desktop\flowmatic`
- Package manager/runtime: `bun`
- Backend stack: NestJS on Fastify
- Frontend stack: Vue
- Smart-city work is real and no longer mock-only in the main flow
- The git worktree is already dirty. Do not assume every unstaged change was created by the current agent. Read before changing.

## 2. Major implemented state

### Authentication and organizations

Implemented:
- real auth and session flow
- logout flow
- profile editing and profile/settings page
- organization CRUD
- organization switching
- organization invitations
- accept/decline invitation flow
- max 3 organizations per user
- registration auto-creates `"<User name>'s Organization"`

Relevant areas:
- `backend/src/modules/auth/`
- `frontend/src/pages/settings-page.vue`

### Backend modernization and cleanup

Implemented:
- Fastify migration
- stronger DTO validation in several backend modules
- CUID param pipe
- response-shape cleanup in several endpoints
- config cleanup for generation timeout
- queue typing cleanup
- shared CSV utilities and shared data typing
- security/auth cleanup on previously weak endpoints
- unused backend dependency pruning

Relevant areas:
- `backend/src/main.ts`
- `backend/src/common/`
- `backend/src/modules/pipeline/`
- `backend/src/modules/export/`
- `backend/src/modules/quality/`
- `backend/src/modules/cleaning/`

### Smart-city pipelines

Implemented:
- smart-city pipeline CRUD
- source CRUD
- source simulation
- real external WebSocket source connection
- real external HTTP polling source connection
- live event flow into processing
- generated n8n-style workflow graph
- processing-unit model selection
- processing test endpoint

Relevant areas:
- `backend/src/modules/smart-city/`
- `frontend/src/pages/connectors-page.vue`

### Data lake and medallion storage

Implemented:
- S3-compatible data lake CRUD
- test/disconnect/delete
- bucket auto-create if missing
- medallion storage tiers:
  - `raw`
  - `cleaned`
  - `business`
- smart-city raw events write into lake
- smart-city processing outputs write into lake
- in-app lake browser
- replay/backfill from historical `sensorEvent` records into medallion storage

Relevant areas:
- `backend/src/modules/storage/storage.service.ts`
- `backend/src/modules/smart-city/smart-city.service.ts`
- `frontend/src/pages/connectors-page.vue`

### Static exports via adapters

Implemented:
- reuse of existing export adapters for smart-city pipeline stages
- manual export of `raw`, `cleaned`, `business`
- adapters:
  - JSON
  - CSV
  - PostgreSQL
  - MongoDB
  - Hugging Face
- encrypted saved credentials for supported adapters
- export target persistence
- continuous export targets
- queue-backed target execution
- export run history
- structured adapter forms in UI
- advanced JSON override remains available

Relevant areas:
- `backend/src/modules/export/`
- `backend/src/modules/smart-city/`
- `frontend/src/api/client.ts`
- `frontend/src/pages/connectors-page.vue`

### Research/model assets

Implemented:
- local `models/` workspace exists
- research checkpoints/assets exist locally
- research model selection in processing unit
- local model training run support

Relevant areas:
- `models/`
- `backend/src/modules/smart-city/smart-city.service.ts`

### Federated learning integration

Implemented:
- coordinator connect/test/disconnect
- HTTP and WebSocket support
- forwarding of processing/training updates
- persisted federated state in pipeline stream config
- real round lifecycle in app state:
  - start round
  - submit local update
  - aggregate round
  - sync global model state
- round ledger in UI
- registration/global model/current round surfaced in UI

Important note:
- this is a generic coordinator contract, not a vendor-specific protocol lockstep implementation

Relevant areas:
- `backend/src/modules/smart-city/smart-city.service.ts`
- `backend/src/modules/smart-city/smart-city.controller.ts`
- `frontend/src/pages/connectors-page.vue`
- `frontend/src/api/client.ts`

### Observability

Implemented:
- smart-city observability summary endpoint
- operations snapshot on connectors page
- computed alerts for:
  - no sources
  - no running sources
  - source errors
  - export failures
  - federated errors

Relevant areas:
- `backend/src/modules/smart-city/smart-city.service.ts`
- `backend/src/modules/smart-city/smart-city.controller.ts`
- `frontend/src/pages/connectors-page.vue`

## 3. Connectors page meaning today

The page now represents one pipeline:

1. Sources  
   Live data ingress. Simulated or external WS/HTTP polling sources.

2. Runtime Processing Unit  
   Model used on live incoming data. Processing tests and active-model selection live here.

3. Data Lake & Export  
   Medallion lake storage, lake browser, replay/backfill, static adapter exports, continuous export targets, export run history.

4. Federated Model Training  
   Coordinator connection, round lifecycle, global model sync, round ledger, local training fallback.

Main file:
- `frontend/src/pages/connectors-page.vue`

## 4. Migrations and generated state

Important:
- Prisma migration files were added for organizations, invitations, export credentials, smart-city pipelines/models, and smart-city export targets.
- If a fresh environment does not have the latest schema, backend features will not fully work until migrations are applied.

Check:
- `backend/prisma/migrations/`

Likely required on a fresh environment:
```powershell
cd C:\Users\BG\Desktop\flowmatic\backend
bun run prisma:migrate
```

## 5. What has been validated

These checks passed during the recent work:

Backend:
- `bun run typecheck`
- `bun run build`

Frontend:
- `bun run type-check`
- `bun run build-only`

Note:
- frontend `build-only` needed elevated spawn permission for Vite/esbuild in this environment

## 6. What is still left

The core product shape exists. What remains is mostly production hardening and deeper operational work.

### Highest-priority remaining work

1. Delivery reliability
- retry/backoff policy for exports and federated updates
- dead-letter handling
- stronger idempotency/dedup semantics
- explicit delivery audit trail

2. Better observability
- real charts, not just snapshot cards
- throughput and latency trends
- per-source health history
- export failure trends over time
- federated round metrics over time

3. Alert persistence
- alerts history
- acknowledge/resolve flow
- configurable thresholds
- optional outbound notifications

4. Coordinator-specific federated parity
- if a real coordinator has a strict protocol, align exact payloads, endpoints, signatures, checkpoint rules, and aggregation semantics

5. Processing-unit hardening
- schema/model compatibility checks
- rollback/version pinning
- richer runtime metrics

6. Data lake maturity
- compaction/batching
- manifest/catalog metadata
- retention policies
- broader replay by time window and stage strategy

7. Export UX polish
- edit/delete export targets in UI
- better presets/templates
- stronger preview/validation UX

8. Test coverage
- backend integration tests for smart-city flows
- external source tests
- MinIO/S3 write tests
- federated round tests
- frontend interaction tests for connectors page

## 7. Good next steps for another agent

If another agent picks this up, the safest next sequence is:

1. read `CURRENT_STATE_HANDOFF.md`
2. inspect `backend/src/modules/smart-city/`
3. inspect `frontend/src/pages/connectors-page.vue`
4. run:
   - `cd backend && bun run typecheck`
   - `cd backend && bun run build`
   - `cd frontend && bun run type-check`
5. verify whether Prisma migrations are already applied
6. only then continue with:
   - reliability/delivery hardening
   - observability depth
   - alert persistence
   - coordinator-specific federated contract work

## 8. Files most likely to matter first

Backend:
- `backend/src/modules/smart-city/smart-city.service.ts`
- `backend/src/modules/smart-city/smart-city.controller.ts`
- `backend/src/modules/export/export.service.ts`
- `backend/src/modules/storage/storage.service.ts`
- `backend/prisma/schema.prisma`

Frontend:
- `frontend/src/pages/connectors-page.vue`
- `frontend/src/api/client.ts`
- `frontend/src/pages/settings-page.vue`
- `frontend/src/modules/auth/pages/pipelines-page.vue`

## 9. Practical caution

- Do not assume this worktree is clean.
- Do not revert unrelated user changes.
- Do not switch away from Bun.
- Do not reintroduce mock behavior into paths that are already real.
