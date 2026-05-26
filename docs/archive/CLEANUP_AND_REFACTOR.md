# Flowmatic - Cleanup and Refactor Guide

---

## 1. Duplicate CSV Parsing Logic (HIGH PRIORITY)

CSV parsing is implemented **three separate times** across the codebase:

| Location | File | Lines |
|----------|------|-------|
| IngestionService | `src/modules/ingestion/ingestion.service.ts` | 35-58 |
| PipelineService | `src/modules/pipeline/pipeline.service.ts` | 239-285 |
| ExportService | `src/modules/export/export.service.ts` | 199-245 |

All three implement their own CSV line parser with quote handling. The `IngestionService` version is the simplest (no quote handling), while `PipelineService` and `ExportService` have nearly identical `parseCsvLine()` methods.

**Fix:** Extract a shared `CsvParser` utility class into `src/common/utils/csv-parser.util.ts`:

```typescript
// src/common/utils/csv-parser.util.ts
export interface DataRow {
  [key: string]: unknown
}

export function parseCsvBuffer(buffer: Buffer): { rows: DataRow[]; columns: string[] } {
  const text = buffer.toString('utf-8').trim()
  if (!text) return { rows: [], columns: [] }

  const lines = text.split(/\r?\n/).filter(l => l.length > 0)
  if (lines.length === 0) return { rows: [], columns: [] }

  const columns = parseCsvLine(lines[0])
  const rows: DataRow[] = []

  for (let i = 1; i < lines.length; i++) {
    const values = parseCsvLine(lines[i])
    if (values.length === 0) continue
    const row: DataRow = {}
    columns.forEach((col, idx) => { row[col] = values[idx] ?? '' })
    rows.push(row)
  }

  return { rows, columns }
}

export function parseCsvLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"'
        i++
      } else {
        inQuotes = !inQuotes
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current)
      current = ''
    } else {
      current += char
    }
  }
  result.push(current)
  return result
}
```

Then replace all three implementations with `import { parseCsvBuffer, DataRow } from 'src/common/utils/csv-parser.util'`.

Also extract `dataToCSV` which is duplicated in:
- `PipelineService.processPipelineJob()` (lines 468-481)
- `CSVExportAdapter.dataToCSV()` (lines 72-94)
- `HuggingFaceExportAdapter.dataToCSV()` (lines 156-174)

---

## 2. Duplicate CSV Escape Logic (HIGH PRIORITY)

CSV escaping is implemented **three times**:

| Location | Method |
|----------|--------|
| `PipelineService.processPipelineJob` | inline lambda (line 475) |
| `CSVExportAdapter.escapeCSV` | `file.adapters.ts:96-101` |
| `HuggingFaceExportAdapter.escapeCSV` | `huggingface.adapter.ts:179-184` |

**Fix:** Add to the shared CSV utility:

```typescript
export function escapeCsvValue(value: string, delimiter: string = ','): string {
  if (value.includes(delimiter) || value.includes('"') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`.replace(/\n/g, '\\n')
  }
  return value
}

export function rowsToCsv(
  data: Record<string, unknown>[],
  options?: { delimiter?: string }
): string {
  if (data.length === 0) return ''
  const delimiter = options?.delimiter ?? ','
  const headers = Object.keys(data[0])
  const headerLine = headers.join(delimiter)
  const dataLines = data.map(row =>
    headers.map(h => escapeCsvValue(String(row[h] ?? ''), delimiter)).join(delimiter)
  )
  return [headerLine, ...dataLines].join('\n')
}
```

---

## 3. Duplicate `DataRow` Interface (HIGH PRIORITY)

The `DataRow` interface is defined in `ingestion.service.ts` and imported by `quality.service.ts`, `cleaning.service.ts`, `pipeline.service.ts`, and `export.service.ts`. This creates a **transitive module dependency** — every module that needs the data type depends on the ingestion module.

**Fix:** Move to a shared types file:

```typescript
// src/common/types/data.types.ts
export interface DataRow {
  [key: string]: unknown
}
```

Then update all imports to use `src/common/types/data.types`.

---

## 4. Duplicate `AuthContext` Interface (MEDIUM PRIORITY)

`AuthContext` is defined **in two places**:

1. `src/modules/auth/auth-context.service.ts:6-10`:
   ```typescript
   export interface AuthContext {
     userId: string
     organizationId: string
     role: string   // <-- loosely typed
   }
   ```

2. `src/common/types/api.types.ts:10-14`:
   ```typescript
   export interface AuthContext {
     userId: string
     organizationId: string
     role: 'admin' | 'member' | 'viewer'  // <-- properly typed
   }
   ```

**Fix:** Delete the one in `api.types.ts`, keep the one in `auth-context.service.ts`, but update it to use the union type:

```typescript
export type UserRole = 'admin' | 'member' | 'viewer'

export interface AuthContext {
  userId: string
  organizationId: string
  role: UserRole
}
```

Export `UserRole` from this file and use it in the Prisma schema comment too.

---

## 5. Duplicate Express `Request` Type Augmentation (MEDIUM PRIORITY)

The Express `Request` type augmentation for `authContext` is declared in **two files**:

1. `src/modules/auth/auth.guard.ts:5-9`
2. `src/modules/pipeline/pipeline.controller.ts:19-23`

**Fix:** Move to a single declaration file:

```typescript
// src/types/express.d.ts
import { AuthContext } from '../modules/auth/auth-context.service'

declare module 'express' {
  interface Request {
    authContext?: AuthContext
  }
}
```

Remove the duplicate `declare module` blocks from both `auth.guard.ts` and `pipeline.controller.ts`.

---

## 6. `process.env` Direct Access vs. `AppConfigService` (MEDIUM PRIORITY)

Several services read environment variables directly instead of using `AppConfigService`:

| File | Line | Direct `process.env` usage |
|------|------|---------------------------|
| `main.ts` | 51 | `process.env.SERVER_PORT` |
| `main.ts` | 45 | `app.get(AppConfigService).security.backendCorsOrigins` (correct) |
| `ingestion.controller.ts` | 69 | `process.env.S3_BUCKET` |
| `pipeline.service.ts` | 174 | `process.env.S3_BUCKET` |
| `pipeline.service.ts` | 343-353 | `process.env.OPENAI_API_KEY`, `process.env.OPENAI_API_KEY` |
| `pipeline.service.ts` | 445 | `process.env.S3_BUCKET` |
| `export.service.ts` | 173 | `process.env.S3_BUCKET` |
| `file.adapters.ts` | 44 | `process.env.S3_BUCKET` |
| `file.adapters.ts` | 143 | `process.env.S3_BUCKET` |
| `storage.service.ts` | 28-31 | `process.env.S3_ACCESS_KEY`, etc. |
| `auth.controller.ts` | 55 | `process.env.NODE_ENV` |

**Fix:** Inject `AppConfigService` where needed and use `config.s3.bucket`, `config.openai.apiKey`, etc. The `StorageService` already takes `PrismaService` as a dependency; it should take `AppConfigService` instead and use it to configure the S3 client.

---

## 7. Duplicate `ConfigModule.forRoot({ isGlobal: true })` Registration (MEDIUM)

Both `app.module.ts` AND `config.module.ts` register `ConfigModule.forRoot({ isGlobal: true })`:

- `src/app.module.ts:33`: `ConfigModule.forRoot({ isGlobal: true })`
- `src/common/config/config.module.ts:7-11`: `NestConfigModule.forRoot({ isGlobal: true, ... })`

This means `ConfigModule` is registered globally **twice**. The one in `app.module.ts` is redundant.

**Fix:** Remove `ConfigModule.forRoot({ isGlobal: true })` from `app.module.ts`. The `AppConfigModule` already handles global registration.

---

## 8. Controllers Using `@Res()` Manual Response Handling (MEDIUM)

The `AuthController` uses `@Res()` and manually calls `res.json()`, `res.status()`, `res.cookie()`. This bypasses NestJS interceptors and serialization:

```typescript
// auth.controller.ts - current pattern
@Post('login')
async login(@Body() body: ..., @Res() res: Response): Promise<void> {
  res.cookie('flowmatic_session', token, { ... })
  res.json({ success: true, user: { ... } })
}
```

**Fix:** Use NestJS return pattern with `@Res({ passthrough: true })` to keep cookie access while still letting NestJS handle the response:

```typescript
@Post('login')
async login(@Body() body: ..., @Res({ passthrough: true }) res: Response) {
  res.cookie('flowmatic_session', token, { ... })
  return { success: true, user: { ... } }
}
```

For error responses, use NestJS exceptions (`BadRequestException`, `NotFoundException`) instead of `res.status(400).json(...)`.

---

## 9. No DTOs for Request Bodies (HIGH)

Controllers accept raw `@Body()` without proper DTO classes:

| Controller | Endpoint | Body Type |
|------------|----------|-----------|
| `AuthController` | `/login` | `{ email: string; password?: string }` inline |
| `AuthController` | `/change-password` | `{ password: string }` inline |
| `PipelineController` | `/runs` | Query params parsed manually |
| `ExportController` | `/runs/:runId/export` | Inline body type |
| `QualityController` | `/analyze` | Inline body type |
| `CleaningController` | `/clean` | Inline body type |

**Fix:** Create proper DTO classes with class-validator decorators:

```typescript
// src/modules/auth/dto/login.dto.ts
import { IsEmail, IsOptional, IsString } from 'class-validator'
import { ApiProperty } from '@nestjs/swagger'

export class LoginDto {
  @ApiProperty()
  @IsEmail()
  email: string

  @ApiProperty({ required: false })
  @IsOptional()
  @IsString()
  password?: string
}
```

```typescript
// src/modules/auth/dto/change-password.dto.ts
import { IsString, MinLength } from 'class-validator'
import { ApiProperty } from '@nestjs/swagger'

export class ChangePasswordDto {
  @ApiProperty()
  @IsString()
  @MinLength(6)
  password: string
}
```

Same pattern for export, quality, and cleaning endpoints.

---

## 10. S3 Bucket Name Hardcoded as Fallback (MEDIUM)

The S3 bucket name `'flowmatic-uploads'` is used as a fallback in **5 places**:

- `ingestion.controller.ts:69`
- `pipeline.service.ts:174, 445`
- `export.service.ts:173`
- `file.adapters.ts:44, 143`

But `AppConfigService.s3.bucket` has `'flowmatic-media'` as its default. These are **two different bucket names**.

**Fix:** Inject `AppConfigService` and use `config.s3.bucket` everywhere. Remove all `process.env.S3_BUCKET || 'flowmatic-uploads'` patterns.

---

## 11. `StorageService` Depends on `PrismaService` Unnecessarily (MEDIUM)

`StorageService` (`storage.service.ts`) injects `PrismaService` but **never uses it**. All methods are pure S3 operations.

**Fix:** Remove the `PrismaService` dependency. The constructor should take `AppConfigService` instead for S3 configuration:

```typescript
constructor(private readonly config: AppConfigService) {
  const { accessKeyId, secretAccessKey, region, accessEndpoint, usePathStyle } = config.s3
  // ... build S3Client from config
}
```

Also remove `PrismaModule` from `StorageModule.imports`.

---

## 12. Legacy / Dead Code (MEDIUM)

Several files contain dead code that should be removed:

### `StorageService` legacy methods (`storage.service.ts:164-171`):
```typescript
createDataset(name: string, description?: string): Promise<any> {
  this.logger.log(`Creating dataset: ${name}`)
  return Promise.resolve({ id: '1', name, description })
}

getDatasets(): Promise<any[]> {
  return Promise.resolve([])
}
```

### `ExportController.generateSampleData` (`export.controller.ts:178-193`):
```typescript
private generateSampleData(totalRows: number, sampleSize: number): Record<string, unknown>[] {
  // ... generates random data but is never called
}
```

### `ExportService.generateMockData` (`export.service.ts:250-262`):
Used as fallback when no result file exists. Should be removed once pipeline always produces results.

### `CommonModule` (`common.module.ts`):
Exports `MediaUrlService` and `QdrantAdapter` but neither is imported by any module.

### `HuggingFaceExportAdapter.getCurrentUser` (`huggingface.adapter.ts:329-332`):
```typescript
private getCurrentUser(): string | null {
  return null  // placeholder that always returns null
}
```

The actual user info is fetched in the `export()` method via `whoAmI()` but the README still uses `'username'` as a placeholder.

### Unused imports in `package.json`:
- `install` (npm package, line 69) - this is the `npm install` command itself, accidentally added as a dependency.
- `expo-server-sdk` - no usage found.
- `ffmpeg-static`, `fluent-ffmpeg` - no usage found.
- `lodash.shuffle` - `array.util.ts` implements Fisher-Yates shuffle instead.
- `sharp` - no usage found.
- `word-extractor` - no usage found.
- `pdf-parse` - no usage found.
- `mammoth` - no usage found.
- `bull` / `@nestjs/bull` - project uses PgBoss/RabbitMQ instead.
- `cors` - NestJS handles CORS natively.
- `langchain` / `@langchain/langgraph` - only `@langchain/openai` and `@langchain/core` are used.

**Fix:** Remove unused dependencies from `package.json`. Remove dead methods.

---

## 13. `any` Type Usage (MEDIUM)

Several places use `any` instead of proper types:

| File | Line | Usage |
|------|------|-------|
| `storage.service.ts` | 164, 169 | `Promise<any>`, `Promise<any[]>` |
| `export.service.ts` | 43 | `PaginatedResponse<Record<string, any>>` |
| `export.controller.ts` | 78, 126 | `Record<string, any>` for settings |
| `postgres.adapter.ts` | 19 | `Record<string, any>` for validate param |
| `mongodb.adapter.ts` | 14 | `Record<string, any>` for validate param |
| `huggingface.adapter.ts` | 20 | `Record<string, any>` for validate param |
| `boss.service.ts` | 38, 47, 57 | `as unknown as { publish: (q: string, d?: any, ...` |
| `pgboss.queue.adapter.ts` | 45 | `(('data' in job) as any)` |
| `api.types.ts` | 2, 41 | `ApiResponse<T = any>`, `Record<string, any>[]` |

**Fix:** Replace with proper types:
- Export adapter validate: `Record<string, unknown>` (already the base type).
- `boss.service.ts`: Use proper PgBoss types or a minimal interface.
- `api.types.ts`: Use `unknown` as the default generic parameter.

---

## 14. Missing Error Handling in `IngestionService` (MEDIUM)

The CSV parser in `IngestionService.ingestCSV()` has no quote handling and will break on any CSV with quoted fields containing commas:

```typescript
// ingestion.service.ts:43 - naive split
const headers = lines[0].split(',').map(h => h.trim())
```

Compare with `PipelineService.parseCsvLine()` which properly handles quoted fields.

**Fix:** Use the shared `parseCsvBuffer` utility from issue #1.

---

## 15. Quality & Cleaning Controllers Have No Auth Guard (LOW but notable)

`QualityController` and `CleaningController` endpoints are completely unauthenticated:

```typescript
// quality.controller.ts
@Controller('quality')  // no @UseGuards(AuthGuard)
export class QualityController {
  @Post('analyze')
  analyzeQuality(@Body('data') data: DataRow[], @Body('columns') columns: string[]) {
```

This means anyone can call these endpoints without authentication. If these are meant to be internal-only (called by PipelineService), they should either:
- Be removed as controllers and only used as services, or
- Have auth guards applied.

**Fix:** Either add `@UseGuards(AuthGuard)` or remove the controllers and keep only the services (which is what `PipelineService` uses directly).

---

## 16. `QualityService.detectOutliers` Recomputes Mean/StdDev Per Row (HIGH performance)

The outlier detection recomputes column mean and standard deviation **for every single row**:

```typescript
// quality.service.ts:121-135
for (const row of data) {             // O(n)
  for (const col of numericColumns) { // O(m)
    const columnValues = data.map(r => Number(r[col])).filter(v => !isNaN(v)) // O(n) !!
    const mean = columnValues.reduce((a, b) => a + b, 0) / columnValues.length // O(n) !!
    // ...
  }
}
```

This is **O(n^2 * m)** when it should be **O(n * m)**.

**Fix:** Pre-compute mean and stdDev per column once:

```typescript
private detectOutliers(data: DataRow[], numericColumns: string[], threshold = 3) {
  // Pre-compute stats per column
  const stats = new Map<string, { mean: number; stdDev: number }>()
  for (const col of numericColumns) {
    const values = data.map(r => Number(r[col])).filter(v => !isNaN(v))
    if (values.length < 2) continue
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const variance = values.reduce((a, v) => a + Math.pow(v - mean, 2), 0) / values.length
    const stdDev = Math.sqrt(variance)
    if (stdDev > 0) stats.set(col, { mean, stdDev })
  }

  const outlierRows: DataRow[] = []
  const outlierColumns = new Set<string>()

  for (const row of data) {
    let isOutlier = false
    for (const col of stats.keys()) {
      const value = Number(row[col])
      if (isNaN(value)) continue
      const { mean, stdDev } = stats.get(col)!
      if (Math.abs((value - mean) / stdDev) > threshold) {
        isOutlier = true
        outlierColumns.add(col)
      }
    }
    if (isOutlier) outlierRows.push(row)
  }

  return { count: outlierRows.length, rows: outlierRows, columns: Array.from(outlierColumns) }
}
```

The same issue exists in `CleaningService.handleOutliers` — it recomputes mean/stdDev per column but at least does it once per column (O(n*m)), not per row.

---

## 17. `ChatOpenAI` Instantiated Per Request (MEDIUM performance)

In `PipelineService.generateLlmSummary()`:

```typescript
// pipeline.service.ts:350-354
const chat = new ChatOpenAI({
  modelName: 'gpt-4o',
  temperature: 0.2,
  openAIApiKey: process.env.OPENAI_API_KEY,
})
```

A new `ChatOpenAI` instance is created every time a summary is generated. This creates a new HTTP client on each invocation.

**Fix:** Create the `ChatOpenAI` instance once in the constructor:

```typescript
constructor(
  // ... other deps
  private readonly config: AppConfigService,
) {
  if (config.openai.apiKey) {
    this.chatModel = new ChatOpenAI({
      modelName: 'gpt-4o',
      temperature: 0.2,
      openAIApiKey: config.openai.apiKey,
    })
  }
}
```

---

## 18. Session Token Generation Is Weak (SECURITY)

In `AuthContextService.createSession()`:

```typescript
// auth-context.service.ts:42
const token = Buffer.from(`${userId}:${Date.now()}:${Math.random()}`).toString('base64')
```

This is **not cryptographically secure**:
- `Math.random()` is predictable.
- Base64 encoding is not encryption.
- The token format is guessable (userId:timestamp:random).

**Fix:** Use `crypto.randomBytes`:

```typescript
import { randomBytes } from 'crypto'

const token = randomBytes(48).toString('hex')
```

---

## 19. `deleteAccount` Deletes Entire Organization (LOGIC BUG)

In `AuthContextService.deleteAccount()`:

```typescript
// auth-context.service.ts:116-128
async deleteAccount(userId: string, organizationId: string): Promise<void> {
  const user = await this.prisma.user.findFirst({
    where: { id: userId, organizationId },
  })
  if (user) {
    await this.prisma.organization.delete({
      where: { id: organizationId },
    })
  }
}
```

Deleting a single user deletes the **entire organization** and all its data (cascade delete). If an org has multiple users, this would destroy everyone's data.

**Fix:** Either:
1. Only allow org deletion if the user is the last member, or
2. Only delete the user, not the org (and assign a new admin if needed).

---

## 20. Inconsistent Response Format (LOW)

API responses are inconsistent:

- Most endpoints return `{ success: true, data: ... }`
- Auth login returns `{ success: true, user: { ... } }` (no `data` wrapper)
- Some use `{ success: true, message: ... }` without `data`
- Error responses vary: `{ success: false, error: ... }` vs NestJS default `{ statusCode, message, error }`

**Fix:** Standardize with a response interceptor:

```typescript
@Injectable()
export class TransformInterceptor<T> implements NestInterceptor<T, ApiResponse<T>> {
  intercept(context: ExecutionContext, next: CallHandler): Observable<ApiResponse<T>> {
    return next.handle().pipe(map(data => ({ success: true, data })))
  }
}
```

Register globally in `main.ts`:
```typescript
app.useGlobalInterceptors(new TransformInterceptor())
```

---

## 21. Missing Input Validation in Pipeline Controller (MEDIUM)

`PipelineController` parses query parameters manually:

```typescript
// pipeline.controller.ts:59-60
const limitNum = limit ? Math.min(parseInt(limit, 10), 100) : 50
const offsetNum = offset ? parseInt(offset, 10) : 0
```

If `limit` or `offset` are non-numeric strings, `parseInt` returns `NaN`, which propagates through Prisma queries.

**Fix:** Create a `ListRunsQueryDto`:

```typescript
export class ListRunsQueryDto {
  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(1)
  @Max(100)
  limit?: number

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(0)
  offset?: number

  @IsOptional()
  @IsIn(['queued', 'processing', 'completed', 'failed'])
  status?: string
}
```

---

## 22. Run ID Validation Is Weak (LOW)

```typescript
// pipeline.controller.ts:76, 93, 110
if (!id || id.length < 5) {
  throw new BadRequestException('Invalid run ID')
}
```

CUID IDs generated by Prisma are always 25+ characters and start with `c`. The length check of 5 is too permissive.

**Fix:** Use a regex matching the CUID format:

```typescript
const CUID_REGEX = /^c[a-z0-9]{20,30}$/
if (!id || !CUID_REGEX.test(id)) {
  throw new BadRequestException('Invalid run ID')
}
```

---

## 23. `ExportService.loadRunData` Falls Back to Mock Data Silently (MEDIUM)

When S3 download fails, `loadRunData` silently returns mock data:

```typescript
// export.service.ts:191-195
} catch (error) {
  this.logger.error(`Failed to load run data from S3 (key: ${key}): ${errorMessage}`)
  return this.generateMockData(run.rowsIngested || 100)  // silent fallback to fake data
}
```

This means exports could silently export **completely fake data** without the user knowing.

**Fix:** Throw an error instead of returning mock data, or at minimum add a flag in the response indicating data is synthetic.

---

## 24. `ExportController` Injects `PrismaService` Redundantly (LOW)

```typescript
// export.controller.ts:26
constructor(
  private readonly exportService: ExportService,
  private readonly prisma: PrismaService,  // used for access checks that ExportService already does
) {}
```

The controller checks run existence and ownership (lines 87-97, 152-162) but `ExportService` already does the same checks. This is a double-fetch pattern.

**Fix:** Remove `PrismaService` from the controller and rely on the service layer for authorization checks.

---

## 25. Timeout Middleware Applied After `app.listen()` (BUG)

In `main.ts`:

```typescript
await app.listen(process.env.SERVER_PORT ?? 8080)  // line 51 - server starts

// lines 55-59 - middleware applied AFTER server is listening
try {
  app.use(generationTimeoutMiddleware())
} catch (err) {
  console.warn('Could not install generation timeout middleware', err)
}
```

Middleware registered after `app.listen()` may not apply to requests that arrive immediately.

**Fix:** Move the middleware registration **before** `app.listen()`.

---

## 26. Double ConfigModule Registration (MEDIUM)

As mentioned in issue #7, both `app.module.ts` and `config.module.ts` register `ConfigModule.forRoot({ isGlobal: true })`. The one in `app.module.ts` line 33 is redundant.

```typescript
// app.module.ts:33 - REMOVE THIS LINE
ConfigModule.forRoot({ isGlobal: true }),
```

The `AppConfigModule` (imported on line 34) already provides `NestConfigModule.forRoot({ isGlobal: true, ... })`.

---

## 27. `PgBossQueueAdapter.subscribe` Type Assertion Issue (LOW)

```typescript
// pgboss.queue.adapter.ts:45
if (!job || typeof job !== 'object' || !(('data' in job) as any)) {
```

The `as any` cast on `('data' in job)` is a TypeScript hack. The `as any` silences the compiler but doesn't change runtime behavior.

**Fix:** Simply remove the `as any`:
```typescript
if (!job || typeof job !== 'object' || !('data' in job)) {
```

---

## 28. `BossService` Uses Unsafe Type Assertions (LOW)

```typescript
// boss.service.ts:38, 44-47, 56-58
const b = this.boss as unknown as { publish: (q: string, d?: any, o?: any) => Promise<unknown> }
```

Multiple `as unknown as` casts to bypass PgBoss typing issues.

**Fix:** Use PgBoss types directly or create a minimal interface:
```typescript
interface BossPublishOptions { startAfter?: Date; ... }
```

---

## 29. `StorageModule` Missing `AppConfigModule` Import (MEDIUM)

`StorageService` reads env vars directly (`process.env.S3_ACCESS_KEY`, etc.) but doesn't import `AppConfigModule`. This works because `ConfigModule` is global, but it's fragile and inconsistent with the rest of the codebase.

**Fix:** Inject `AppConfigService` and use `config.s3.*` properties.

---

## 30. `QualityController` and `CleaningController` Accept Arbitrary Data in Body (HIGH)

These endpoints accept `@Body('data')` as `DataRow[]` without any validation:

```typescript
@Post('analyze')
analyzeQuality(@Body('data') data: DataRow[], @Body('columns') columns: string[]) {
```

A malicious user could send a 100MB JSON array and it would be parsed without limits.

**Fix:** Add DTOs with size limits, or consider removing these public-facing endpoints entirely since `PipelineService` calls the services directly.

---

## Summary Priority Matrix

### Must Fix (Security / Bugs)

| # | Issue | Impact |
|---|-------|--------|
| 18 | Weak session tokens | Security vulnerability |
| 19 | Deleting user deletes entire org | Data loss for multi-user orgs |
| 25 | Middleware after listen | Middleware may not apply |
| 23 | Silent mock data fallback | Fake data exported silently |

### Should Fix (Code Quality)

| # | Issue | Impact |
|---|-------|--------|
| 1 | Triplicate CSV parsing | Maintenance burden, bugs |
| 2 | Triplicate CSV escaping | Maintenance burden |
| 3 | DataRow in wrong module | Circular dependency risk |
| 4 | Duplicate AuthContext | Type inconsistency |
| 5 | Duplicate type augmentation | Compile errors risk |
| 6 | process.env direct access | Inconsistent config |
| 7 | Double ConfigModule registration | Confusion |
| 9 | No DTOs | No validation on inputs |
| 16 | O(n^2) outlier detection | Performance bottleneck |
| 17 | LLM client per request | Resource waste |

### Nice to Have (Polish)

| # | Issue | Impact |
|---|-------|--------|
| 8 | Manual @Res() handling | Inconsistent patterns |
| 10 | Hardcoded bucket name | Config drift |
| 11 | Unused PrismaService dep | Dead dependency |
| 12 | Dead code / unused deps | Bundle size, confusion |
| 13 | `any` types | Type safety |
| 14 | No quote handling in ingestion | CSV parsing bugs |
| 15 | Unauthenticated endpoints | Security policy |
| 20 | Inconsistent response format | API ergonomics |
| 21 | Missing input validation | Robustness |
| 22 | Weak ID validation | Edge cases |
| 24 | Double-fetch pattern | Performance |
| 27-28 | Type assertion hacks | Type safety |
| 29 | Missing config module import | Consistency |
| 30 | No body size limits | DoS risk |
