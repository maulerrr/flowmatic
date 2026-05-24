# Flowmatic - Project Guide

## What Is Flowmatic?

Flowmatic is an **Intelligent Data Preparation Platform** — a backend-as-a-service that automates the dirty work of cleaning, validating, and exporting raw tabular data. Users upload messy CSV/JSON files, Flowmatic runs an automated quality-and-cleaning pipeline, and then exports the polished dataset to any destination (PostgreSQL, MongoDB, Hugging Face, CSV, JSON, or future targets).

Think of it as **"CI/CD for data quality"** — but with an AI copilot that summarizes what it found and recommends next steps.

---

## Problem Statement

Every data scientist, analyst, and ML engineer wastes hours on the same grind:

1. **Ingest** a CSV / JSON dump from a client, partner, or public source.
2. **Profile** the data -- find missing values, duplicates, wrong types, outliers.
3. **Clean** it -- remove duplicates, impute missing values, cap outliers.
4. **Export** the cleaned dataset to a database, cloud bucket, or model-training pipeline.

Flowmatic autom steps 2-4 in a single upload-then-processed-result flow, backed by an async job queue and an LLM-powered summary that explains what was done and why.

---

## Architecture Overview

```
Frontend          Ingestion           Pipeline            Quality +            Export
--------        ----------       ------------       ------------       ------------
React App          Upload +           Async Job         Cleaning         Multi-
                 (React)           Parse)               Queue)           Engine)         Adapter)
```

Frontend             (Upload +             Async Job               Cleaning Engine           Multi-adapter Export system)
Auth (session)     Prisma (PostgreSQL)     S3 (MinIO)     Queue (PgBoss/RabbitMQ)
```

### Technology Stack

| Layer            | Technology                          |
| ---------------- | ----------------------------------- |
| Framework        | NestJS 11 (TypeScript)              |
| Database         | PostgreSQL via Prisma ORM 6         |
| File Storage     | S3 / MinIO via AWS SDK v3           |
| Job Queue        | PgBoss (default) or RabbitMQ        |
| LLM Integration  | OpenAI GPT-4o via LangChain         |
| Auth             | Cookie-based session tokens (bcrypt)|
| API Docs         | Swagger (OpenAPI) at `/api`         |
| Logging          | Pino (pretty-print in dev)          |
| Validation       | class-validator + class-transformer  |

---

## Core Data Flow

### 1. Upload and Ingestion

- User uploads a CSV or JSON file via `POST /api/v1/ingestion/upload`.
- The file is stored in S3 (MinIO locally) and a `StorageFile` record is PostgreSQL.
- A `PipelineRun` is created with status `queued`, a job is published to the queue.

### 2. Async Pipeline Processing

The `PipelineService` subscri to the `pipeline` queue and processes each job:

1. **Download** source file from S3.
2. **Parse** CSV/JSON into in-memory rows.
3. **Quality Analysis** (`QualityService.analyzeQuality`):
   - Missing values per column.
   - Duplicate rows.
   - Column classification (numeric vs. categorical).
   - Outlier detection via Z-score (threshold=3).
4. **Cleaning** (`CleaningService.clean`):
   - Remove duplicate rows.
   - Impute numeric columns with column mean.
   - Impute categorical columns with column mode.
   - Cap outliers to mean +/- 3 sigma.
5. **Result Upload** - cleaned CSV written to S3.
6. **LLM Summary** (`generateLlmSummary`):
   - Sends pipeline stats to OpenAI GPT-4o.
   - Returns structured JSON: overview, scores (initial/final), insights, recommendation.
   - Falls back to rule-based summary if OpenAI unavailable.
7. **Update** `PipelineRun` to `completed` or `failed`.

### 3. Export

- User calls `POST /api/v1/exports/runs/:runId/export` with adapter type and settings.
- The adapter loads cleaned data from S3 (or mock data).
- Export to PostgreSQL, MongoDB, HuggingFace, CSV, or JSON.
- Export record stored in `PipelineExport` table.
- Export history available via `GET /api/v1/exports/runs/:runId/history`.

### 4. Analytics & Monitoring

- `GET /api/v1/pipelines/analytics/summary` - aggregate stats (total, completed, failed, in-progress, success rate, avg processing time).
- `GET /api/v1/pipelines/analytics/charts` - time-series data (uploads per day, status distribution.

- `POST /api/v1/pipelines/cleanup` - bulk-delete old runs.

---

## Database Schema

### Organizations ( Multi-tenancy )

```
Organization {
  id        String   @id @default(cuid())
  name      String
  slug      String   @unique
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  users       User[]
  pipelines   PipelineRun[]
  storageFiles StorageFile[]
}
```

### Users

```
User {
  id             String   @id @default(cuid())
  email          String   @unique
  password       String?   // nullable (SSO users may not have password)
  displayName    String?
  organizationId String
  role           String   @default("member") // admin, member, viewer
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  organization Organization @relation(...)
}
```

### Sessions

```
Session {
  id        String   @id @default(cuid())
  userId    String
  token     String   @unique @db.VarChar(500)
  expiresAt DateTime
  createdAt DateTime @default(now())
}
```

### Pipeline Runs

```
PipelineRun {
  id               String   @id @default(cuid())
  organizationId   String
  status           String   @default("queued") // queued, processing, completed, failed
  sourceFileName   String
  sourceFileId     String?
  rowsIngested     Int      @default(0)
  rowsCleaned      Int      @default(0)
  rowsErrors       Int      @default(0)
  processingTimeMs Int      @default(0)
  jobId            String?  @unique
  errorMessage     String?
  summary          String?  @db.Text
  resultFileId     String?
  resultFileSize   Int?
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  organization     Organization @relation(...)
  sourceFile       StorageFile? @relation("source")
  resultFile       StorageFile? @relation("result")
  exports          PipelineExport[]
}
```

### Storage Files

```
StorageFile {
  id             String   @id @default(cuid())
  organizationId String
  fileName       String
  fileSize       Int
  mimeType       String
  s3Key          String   @unique
  sourceRuns     PipelineRun[] @relation("source")
  resultRuns     PipelineRun[] @relation("result")
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  organization   Organization @relation(...)
}
```

### Pipeline Exports

```
PipelineExport {
  id               String   @id @default(cuid())
  pipelineRunId    String
  adapterType      String  // postgres, mongodb, huggingface, csv, json, parquet
  destination      String  // DB name, repo URL, etc.
  recordsExported  Int      @default(0)
  metadata         Json     @default("{}")
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  pipelineRun      PipelineRun @relation(...)
}
```

---

## API Reference

| Method | Endpoint | Auth | Description |
| ------ | -------- | ---- | ----------- |
| POST | `/auth/login` | No | Email-based auth, returns session cookie |
| GET | `/auth/profile` | Yes | Get user profile |
| POST | `/auth/logout` | Yes | Invalidate session, clear cookie |
| POST | `/auth/change-password` | Yes | Update password |
| POST | `/auth/delete-account` | Yes | Delete account + org |
| POST | `/ingestion/upload` | Yes | Upload CSV/JSON file |
| GET | `/pipelines/runs` | Yes | List pipeline runs (paginated) |
| GET | `/pipelines/runs/:id` | Yes | Get single run details |
| GET | `/pipelines/runs/:id/preview` | Yes | Preview processed data |
| GET | `/pipelines/analytics/summary` | Yes | Aggregate stats |
| GET | `/pipelines/analytics/charts` | Yes | Chart data (7d/30d/90d/all) |
| DELETE | `/pipelines/runs/:id` | Yes | Delete run + S3 files |
| POST | `/pipelines/cleanup` | Yes | Bulk-delete old runs |
| GET | `/exports/adapters` | Yes | List available adapters |
| POST | `/exports/runs/:runId/export` | Yes | Export data to destination |
| GET | `/exports/runs/:runId/preview` | Yes | Preview data with pagination |
| POST | `/exports/validate` | No | Validate export config |
| GET | `/exports/runs/:runId/history` | Yes | Export history |
| POST | `/quality/analyze` | No | Analyze data quality (unauthenticated) |
| POST | `/cleaning/clean` | No | Clean data (unauthenticated) |

---

## How This Could Be Expanded

### 1. Proper Authentication System (P0)

Replace the current email-only login with:
- **OAuth2 / OIDC** via Google, GitHub, Microsoft.
- **Invite-based org registration** (admin creates org, invites team members).
- **Role-based access control** (admin/member/viewer) enforced at every endpoint.
- **API key authentication** for external integrations (no cookies).
- **Password login** with bcrypt hashing (already partially implemented).
- **JWT access tokens** alongside session cookies for SPA + API dual use.

### 2. WebSocket Real-Time Progress (P0)

Add a WebSocket gateway (Socket.IO or native WS):
- Push pipeline status updates (`queued` -> `processing` -> `completed`/`failed`).
- Stream quality analysis results row-by-row as they're computed.
- Push LLM summary the moment it's ready.
- Frontend can show a live progress bar without polling.

### 3. Team & Organization Management (P1)

- Organization settings page (name, slug, logo).
- Member invitation flow (email invite -> accept -> role assignment).
- Organization switcher for users belonging to multiple orgs.
- Audit log (who did what, when).

### 4. Data Profiling & Visualization (P1)

Beyond the current quality report, add:
- **Column-level profiling**: data type distribution, unique value counts, min/max/avg/median for numerics.
- **Correlation matrix**: which columns are correlated.
- **Interactive charts**: histogram per column, box plots for outliers.
- **Data type recommendations**: "Column X looks like a date but 12% fail to parse."

### 5. User-Configurable Cleaning Rules (P2)

Instead of hardcoded clean/mean/mode/cap, allow users to:
- Choose imputation strategy per column (mean, median, mode, forward-fill, drop).
- Set outlier thresholds per column.
- Apply regex-based transformations (e.g., extract ZIP from address).
- Define custom validation rules (reject if `age < 0`).
- Chain multiple cleaning steps into a custom pipeline.
- Save rule presets for reuse.

### 6. Streaming Large File Support (P2)

Current approach loads entire file into memory. For production:
- **Stream-parse** CSV row-by-row (use `csv-parse` or custom streaming parser).
- **Chunk-based processing** (process 1000 rows at a time, stream results back).
- **File size limit** + user notification to upgrade plan.
- **Resumable uploads** for files > 100MB.

### 7. Multi-Format Support (P2)

Extend `IngestionService` to handle:
- **Excel** (.xlsx, .xls) - `excel.util.ts` already exists.
- **Parquet** - efficient columnar format.
- **Google Sheets** - via Google Sheets API.
- **Database connections** - direct PostgreSQL / MySQL query exports.
- **API endpoints** - fetch JSON from REST APIs.

### 8. Schema Drift Detection (P3)

When re-running pipelines on updated versions of the same file:
- **Compare** new schema vs. previous schema.
- **Flag** added/removed columns, type changes.
- **Suggest** migration strategy (remap, drop, add columns).
- **Version** datasets (track schema evolution over time).

### 9. Scheduled / Recurring Pipelines (P2)

Add cron-like scheduling:
- Users schedule daily/weekly/monthly pipeline runs.
- Auto-ingest from S3 prefix (watch for new files).
- Trigger cleaning when upstream data changes.
- Email/Slack notification on completion or failure.
- `@nestjs/schedule` is already imported in `AppModule`.

### 10. Data Versioning & Lineage (P3)

Track data provenance:
- **Version** each pipeline run's output.
- **Diff** between versions (what changed row-by-row).
- **Rollback** to a previous version.
- **Lineage graph** (input file -> pipeline run -> export destination).

### 11. AI Copilot Enhancements (P4)

Beyond the current summary generation:
- **Natural language cleaning rules**: "Remove all rows where the email column doesn't look valid."
- **Auto-suggest cleaning strategies** based on data patterns.
- **Anomaly explanation**: "These 50 rows are outliers because their purchase amount is 10x the average."
- **Data quality chat**: Ask questions about the data, get SQL-like answers.

### 12. More Export Destinations

Add adapters for:
- **Google BigQuery** - large-scale analytics.
- **Amazon S3 ( Athena )** - serverless query.
- **Snowflake** - data warehouse.
- **Kaggle** - ML competition datasets.
- **GitHub** - version-controlled data.
- **Email** - send cleaned file as attachment.

### 13. Quality Scoring Dashboard (P1)

A comprehensive dashboard showing:
- Per-dataset quality score (0-100).
- Historical trend (quality improving/degrading over time).
- Comparison across datasets.
- Bottleneck identification (which cleaning steps take the most time).

### 14. Automated Data Validation

Beyond cleaning, add **schema validation**:
- JSON Schema / Zod schema inference from data.
- Validate incoming data against inferred schema.
- Generate validation reports (pass/fail per row).
- Suggest schema fixes.

### 15. Vector Search for Similar Datasets

The `QdrantAdapter` is already built but unused. Enable:
- Embed dataset profiles (column names, types, sample values) as vectors.
- Store in Qdrant.
- "Find datasets similar to mine" search.
- Power recommendations: "Users who cleaned datasets like yours also applied these rules..."

### 16. Marketplace / Template Library

Create a community-driven library of:
- Cleaning templates (e.g., "US Address Standardizer", "E-commerce Product Catalog Cleaner").
- Validation rule sets (e.g., "GDPR Compliance Check").
- Export configurations.
- Users can share, fork, and rate templates.

### 17. API for External Integrations

Expose a public REST API (with API keys, not cookies) for:
- Programmatic uploads from ETL tools (Airflow, dbt, Prefect).
- Webhook notifications on pipeline completion.
- Batch processing endpoints.

### 18. Data Anonymization / PII Detection

Auto-detect personally identifiable information:
- Email addresses, phone numbers, SSNs, credit card numbers.
- Mask, hash, or remove PII columns.
- Compliance reports (GDPR, CCPA, HIPAA).

### 19. Real-Time Collaboration

Multiple users in the same org working on the same pipeline:
- Collaborative rule editing (like Google Docs for data rules).
- Comments and annotations on specific rows/columns.
- Activity feed.

### 20. Embeddable Widget

Provide a JavaScript widget that partners can embed in their apps:
- Handles upload, shows progress, returns cleaned data.
- White-label option for enterprise customers.

---

## Beta Implementation Priority

| Phase | Features                                    | Est. Effort |
| ----- | ------------------------------------------- | ----------- |
| P0    | OAuth2 login, RBAC, WebSocket progress      | 2-3 weeks   |
| P1    | Team management, data profiling endpoint    | 1-2 weeks   |
| P2    | User-configurable transform rules, scheduled pipelines | 2-3 weeks |
| P3    | Schema drift detection, audit log           | 1-2 weeks   |
| P4    | AI copilot MVP, streaming large files       | 3-4 weeks   |
| P5    | API keys, webhooks, public API              | 1-2 weeks   |

---

## Directory Map

```
src/
├── main.ts                          # Bootstrap, global pipes, Swagger, CORS
├── app.module.ts                    # Root module wiring
├── prisma/
│   ├── prisma.module.ts             # Global Prisma module
│   └── prisma.service.ts            # Prisma client with lifecycle hooks
├── common/
│   ├── config/                      # AppConfigService (typed env vars)
│   ├── adapters/
│   │   ├── qdrant.adapter.ts        # Vector DB (future use)
│   │   └── whatsapp-built-in.adapter.ts  # WhatsApp API (future use)
│   ├── middleware/
│   │   ├── security.middleware.ts   # Host header validation
│   │   └── generation-timeout.middleware.ts  # Extended timeouts
│   ├── pipes/
│   │   └── parse-json.pipe.ts       # JSON string to object pipe
│   ├── queue/
│   │   ├── boss.module.ts           # PgBoss global module
│   │   ├── boss.service.ts          # PgBoss wrapper
│   │   ├── queue.module.ts          # Dynamic: RabbitMQ or PgBoss
│   │   ├── queue.tokens.ts          # QueueClient interface
│   │   └── adapters/
│   │       ├── rabbitmq.queue.adapter.ts
│   │       └── pgboss.queue.adapter.ts
│   ├── s3/
│   │   ├── s3.module.ts
│   │   └── s3.service.ts            # S3 upload/download/delete
│   ├── services/
│   │   └── media-url.service.ts     # S3 URL builder
│   ├── types/
│   │   ├── api.types.ts             # Shared API response types
│   │   └── learner-request.interface.ts
│   ├── utils/                       # Pagination, validation, formatting, OTP
│   └── common.module.ts             # Shared module
├── modules/
│   ├── auth/                        # Session auth, guard, context
│   ├── ingestion/                   # File upload + CSV/JSON parsing
│   ├── pipeline/                    # Async job orchestration + LLM summary
│   ├── quality/                     # Missing, duplicates, outliers detection
│   ├── cleaning/                    # Dedup, impute, cap outliers
│   ├── storage/                     # S3 operations + key generation
│   └── export/                      # Multi-adapter export system
│       ├── adapters/
│       │   ├── base.adapter.ts      # Abstract base class
│       │   ├── registry.ts          # Adapter factory + metadata
│       │   ├── file.adapters.ts     # CSV + JSON adapters
│       │   ├── postgres.adapter.ts
│       │   ├── mongodb.adapter.ts
│       │   └── huggingface.adapter.ts
│       └── types/
│           └── export.types.ts      # Adapter configs + enums
└── types/
    └── bcrypt.d.ts                  # Type augmentation
```
