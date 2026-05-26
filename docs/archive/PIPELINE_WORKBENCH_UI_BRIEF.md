# Flowmatic — Pipeline Workbench UI Brief (Web Application)

Use this document to generate UI references for a **web-based** data platform screen. This is a **responsive web app** (desktop-first, works on tablet/mobile), not a native mobile app. It runs in the browser inside a standard app shell with a left sidebar and scrollable main content area.

---

## Product context

**Product:** Flowmatic — a smart-city / real-time data pipeline platform  
**Screen name:** Pipeline Workbench (sidebar label: “Pipeline Workbench”, route: `/connectors`)  
**Purpose:** Operate one end-to-end pipeline in a single surface:

**Sources → Core Processing Unit → Data Lake & Export → Federated Training**

Users ingest sensor data (simulated or external), preprocess it with an ML model, store medallion-tier outputs in S3-compatible storage, export to external adapters, and optionally coordinate federated learning rounds.

**User persona:** Data engineer / ML ops operator managing urban sensor pipelines (traffic, air quality, weather, etc.)

---

## App shell (surrounding this page)

The page sits inside a **persistent left sidebar layout**:

- **Sidebar navigation items:** Dashboard, **Pipeline Workbench** (this page, badge “live”), Upload, Pipelines, Analytics, Settings
- **Sidebar footer:** User avatar/name/email, Logout
- **Sidebar behavior:** Collapsible on desktop; slide-in drawer on mobile with hamburger menu
- **Main content:** Scrollable area to the right of sidebar
- **Branding:** “Flowmatic” logo with sparkle icon, teal-to-indigo gradient accent

---

## Visual design language (current implementation)

**Theme:** Dark mode primary (deep navy/black backgrounds), with teal/cyan as primary accent and indigo/violet secondary accents.

**Typography:**
- UI text: **Sora** (sans-serif)
- Metrics, logs, endpoints, IDs: **IBM Plex Mono** (monospace)

**Background treatment on this page:**
- Subtle grid overlay
- Soft radial gradients (teal top-left, indigo top-right)
- Rounded cards with glassy/dark panel feel
- Thin borders with low-opacity slate/teal

**Color semantics:**
- **Primary (teal):** Main CTAs, active states, links
- **Success (green):** Running sources, enabled guards
- **Warning (amber):** Stop actions, disconnected WS
- **Destructive (red):** Delete/archive actions, errors
- **Muted chips:** Status pills (WS connected, throughput, lake name)

**Component patterns:**
- Rounded-xl cards and modals
- Pill/chip badges for status
- Segmented toggle buttons (Workbench / Workflow graph)
- Flow-rail cards as stage navigation (4 columns on desktop, 2 on tablet, 1 on mobile)
- Centered modal overlays with blurred dark backdrop (teleported to body, viewport-centered)
- Uppercase micro-labels (10px, wide letter-spacing) for section headers

---

## Page layout (top to bottom)

When a pipeline exists, the page stacks these regions vertically:

1. **Hero header** — pipeline identity + global actions
2. **View toolbar** — mode switch + optional panels
3. **Flow rail** — 4-stage navigation cards
4. **Main content** — one of:
   - Empty state (no pipeline)
   - Workflow graph view
   - Workbench stage panel (Sources / Processing / Lake / Federated)
5. **Optional panels** (toggled off by default):
   - Metrics / observability
   - System log
6. **Modals** (overlay, not inline)

Only **one workbench stage panel is visible at a time** (Sources, Processing, Lake, or Federated). Metrics and Log are optional overlays in the page flow.

---

## 1. Hero header

**Left side:**
- Kicker chip: “Pipeline workbench” + activity icon
- Live status chip: **“WS connected”** (green) or **“WS disconnected/warn”** (amber) — WebSocket connection to live events
- **H1 title:** Selected pipeline name, or fallback “Smart city pipeline”
- **Subtitle line:** `{sources running} · {throughput} GB/s · {lake name or “No lake connected”}`

**Right side — action bar:**

| Control | Type | Behavior |
|--------|------|----------|
| Pipeline selector | Dropdown | Lists all pipelines + “No pipeline selected” |
| **Pipeline** | Primary button (+ icon) | Opens “Create Pipeline” modal |
| **Archive** | Secondary button (trash icon) | Archives current pipeline (only if one selected) |
| **Refresh** | Secondary button | Reloads dashboard data |

---

## 2. View toolbar

Horizontal bar with two groups:

**Left — view mode toggle (segmented control):**

| Tab | Icon | Shows |
|-----|------|-------|
| **Workbench** | Dashboard icon | Stage-based operational UI (default) |
| **Workflow graph** | Workflow icon | Visual 3-column pipeline map |

**Right — optional panels (toggle buttons):**

| Button | Default | Shows |
|--------|---------|-------|
| **Metrics** | Off | Observability snapshot panel |
| **Log** | Off | System log terminal strip |

Active toggles use highlighted pill styling (teal tint).

---

## 3. Flow rail (primary navigation)

Four clickable cards in a row — **this is the main way to switch stages**. Active stage gets a teal border highlight.

### Stage 01 — Sources
- Icon: Radio/antenna (teal)
- Title: **Sources**
- Meta: `{X/Y running} · {sensor mix summary}`
- Click → shows Sources stage

### Stage 02 — Core unit
- Icon: CPU (teal)
- Title: **Core unit**
- Meta: **Active model name** (truncated), or “No model selected”
- Inline link: **Configure** (teal text) → opens Core Unit modal (does not change stage)
- Click card → shows Processing stage

### Stage 03 — Lake & export
- Icon: Database (teal)
- Title: **Lake & export**
- Meta: S3 bucket name or “No output target”
- Click → shows Lake stage

### Stage 04 — Federated
- Icon: Network (teal)
- Title: **Federated**
- Meta: `{connection status} · {N} rounds`
- Click → shows Federated stage

Each card has a thin gradient accent bar at top; hover lifts card slightly.

---

## 4. Empty state (no pipeline selected)

Centered dashed-border panel:
- Large radio icon (teal)
- Heading: **“Create your first real-time pipeline”**
- Subtext about persistence per organization
- Primary button: **Create Pipeline**

---

## 5. Workbench mode — stage panels

### STAGE A: Sources

**Header row:**
- Title: **Sources** + radio icon
- Description: “Connect simulated or external feeds, then start streaming.”
- Buttons:
  - **Add source** (primary) → Connect Source modal
  - **Poll HTTP feeds** (secondary, disabled without pipeline)

**Main layout:** Two columns on XL screens.

**Left — source list cards** (or empty dashed state “No sources connected yet”)

Each source card shows:
- Sensor icon (varies by kind: air quality, traffic video, power, network, weather, parking)
- Name + **status badge** (RUNNING = green pill, else gray)
- Metadata: `{WEBSOCKET|HTTP_POLLING} / {SIMULATED|EXTERNAL} / {sensorKind}`
- Endpoint or poll interval
- Last event time or “No event received yet”
- Error message in red if present

**Per-source action buttons (4-column grid):**

| Button | Style | When |
|--------|-------|------|
| **Start** | Green | Source not running |
| **Stop** | Amber | Source running |
| **Test** | Neutral border | Always |
| **Delete** | Red border | Always |

**Right sidebar cards:**
1. **Source status** — running count, WS status, event count, mix summary
2. **Recent events** — up to 5 event cards with source name, time, sensor type, location

---

### STAGE B: Core processing unit (Processing)

Accent border (teal tint) on panel.

**Header row:**
- Title: **Core processing unit** + CPU icon
- Description: “Validates, cleans, and scores live events before export.”
- Buttons:
  - **Configure core unit** (primary, settings icon) → Core Unit modal
  - **Test sample event** (secondary, disabled if no model selected)

**Three stat cards in a row:**

1. **Active model**
   - Shows current model name/ID
   - Link: **Change model** → Core Unit modal

2. **Throughput**
   - Large number + “GB/s”

3. **Guards**
   - Three pill badges: **Schema**, **Clean**, **Detect**
   - Green border/text when enabled, gray when off

**Optional block:** **Latest output** — scrollable JSON preview of last processing test/result

---

### STAGE C: Data Lake & Export

**Header:**
- Label: “Stage 03”
- Title: **Data Lake & Export** + database icon
- Description about medallion storage + static exports
- **Connect lake** (primary) → S3 Data Lake modal

**Section 1 — Lake connection card**
- Hard drive icon, lake name, bucket
- If connected: **Test**, **Disconnect**, **Delete** buttons

**Section 2 — Info grid (2×3 small cards)**
- Linked lake name/provider
- Output path / base prefix
- Last connection test status + timestamp
- What gets stored: raw / cleaned / business tiers
- Medallion layout explanation

**Section 3 — Lake browser**
- **Refresh browser** button
- Three columns: **raw**, **cleaned**, **business**
- Each shows prefix path + up to 5 object files (name, size, modified date)

**Section 4 — Replay and backfill**
- **Stage** dropdown: All medallion stages / Raw only / Cleaned only / Business only
- **Events** number input (1–5000)
- **Run backfill** button
- Last backfill summary (counts per tier, timestamp)

**Section 5 — Static export (large form)**

Top fields (4-column grid):

| Field | Type | Options |
|-------|------|---------|
| Target name | Text | |
| Stage | Select | Raw / Cleaned / Business |
| Adapter | Select | Dynamic list (JSON file, CSV, PostgreSQL, MongoDB, Hugging Face, etc.) |
| Rows | Number | 1–500 |
| Save credentials | Checkbox | |
| Continuous target | Checkbox | |

**Adapter-specific settings** (changes based on adapter):

- **JSON:** Pretty print checkbox
- **CSV:** Delimiter select (comma, semicolon, tab, pipe)
- **PostgreSQL:** Host, port, database, username, password, table, if-exists (append/replace)
- **MongoDB:** URI, database, collection, if-exists (append/replace)
- **Hugging Face:** Token, repo name, file name, commit message, private dataset checkbox

**Advanced:** Toggle **Show advanced JSON** → textarea for raw adapter JSON  
**Continuous mode:** Cadence seconds input (15–3600)

**Actions:**
- **Save target** (secondary)
- **Export stage** (primary)

**Lists below:**
- **Saved export targets** (up to 6): name, stage/adapter/status, continuous cadence, last run, errors; buttons **Use**, **Run now**, **Pause/Enable continuous**
- **Recent export runs** (up to 6): stage → adapter, status, row counts, destination, errors

---

### STAGE D: Federated Model Training

**Header:**
- Label: “Step 4”
- Title: **Federated Model Training** + network icon
- Note: federated stage does NOT control live preprocessing model (that’s Core Unit)
- **Open workflow graph** button → switches to Workflow graph view

**Left column — Coordinator connection form**

Status display + fields:

| Field | Type | Options/placeholder |
|-------|------|---------------------|
| Protocol | Select | HTTP / WebSocket |
| Endpoint | Text | Demo coordinator URL placeholder |
| Project ID | Text | e.g. astana-q1 |
| Node ID | Text | e.g. flowmatic-node-1 |
| Topic | Text | e.g. smart-city.training |
| API Key | Password | Optional bearer token |

Buttons: **Connect** (primary), **Test**, **Disconnect**

Status cards: Last coordinator test, Last error, Registration ID, Global model version

**Right column — Round protocol**

- **Sync global model** button
- Active round summary + current round ID
- **Round name** + **Sample count** inputs
- **Start round** (primary, full width)
- Local update fields: Checkpoint URI, Update samples, Update notes
- **Submit local update**
- Aggregation fields: Global model version, Aggregated checkpoint, Aggregation summary
- **Aggregate round**

**Local training fallback:**
- **Train baseline** button (trains from Astana CSV dataset)

Summary cards: Latest training run, Workflow graph node count

**Lists:**
- **Round ledger** (up to 4 rounds): name, status, ID, participants, timestamps
- **Recent training runs** (up to 3): name, model type, status, dataset path

---

## 6. Workflow graph view (alternate mode)

Large dark canvas panel (~620px min height, horizontal scroll on narrow screens).

**Header:**
- “Workflow map” label
- Node count from live pipeline state
- **Regenerate** button (teal tinted)

**3-column layout:**

**Column 1 — Data Sources**
- Card per connected source (icon, name, type/status)
- Dashed **+** button → Add source modal

**Column 2 — Core unit (center, clickable)**
- Highlighted card with teal border
- Shows active model name
- Stats: Running sources count, Throughput GB/s
- Click → Core Unit modal

**Column 3 — Outputs**
- **S3 Data Lake** card (bucket status)
- **Federated Training** card (status/protocol)

---

## 7. Optional: Metrics panel (toggle)

When “Metrics” is on, appears above stage content.

**4 KPI cards:**
- Events 24h + last event time
- Sources running + error count
- Export successes/failures + last export time
- Federated status + global model version

**Alerts list:** severity + message, or “No active alerts”

---

## 8. Optional: System log (toggle)

Terminal-style panel at bottom:
- Header: **SYSTEM LOG** + status (`ready` / `syncing`)
- Monospace log lines prefixed with `>`
- Example: `[READY] Create a pipeline to connect sources`
- Empty: “No runtime messages yet.”
- Max height ~12rem, scrollable

---

## 9. Modals (all centered overlays)

Shared pattern: dark blurred backdrop, rounded card, X close, Cancel + primary action.

### Modal 1: Create Pipeline
- **Name** text input (placeholder: “Astana Traffic Intelligence”)
- **Description** textarea
- **Cancel** | **Create**

### Modal 2: Connect Source
- **Name** text input
- **Transport toggle:** WebSocket | HTTP Polling (2-button segmented)
- **Sensor kind** select: Air Quality, Traffic Video, Power Grid, Network, Weather, Parking
- **Mode toggle:** Simulated | External
- **If Simulated:** info box showing resolved sensor simulator endpoint (read-only)
- **If External:** Endpoint URL input (wss:// or https://)
- **Poll interval** number (ms)
- **If External — advanced:**
  - Payload path, Location field
  - WebSocket: optional subscribe message JSON
  - HTTP: GET/POST method select
- **Cancel** | **Connect**

### Modal 3: Connect S3 Data Lake
- **Lake name** (full width)
- **Provider** select: Custom S3, AWS S3, MinIO, Cloudflare R2
- **Bucket**, **Region**, **Endpoint**, **Base prefix**
- **Access key**, **Secret key** (password)
- **Cancel** | **Save**

### Modal 4: Core Unit (most important for ML ops)
- Title: **Core Unit** + CPU icon

**Processing model** (dropdown):
- Option: “No model selected”
- Optgroup **Research checkpoints** (from local model registry)
- Optgroup **Trained artifacts** (from pipeline training runs)
- Each option shows name + detail (kind/dataset or status/version)
- Empty state helper text if no models

**Pipeline guards** (checkbox rows):
- Anomaly Detection
- Schema Validation
- Auto Cleaning

**Throughput limit:**
- Range slider 1–20 GB/s (step 0.5)
- Live value label in monospace teal

**Cancel** | **Apply** (save icon, disabled while saving)

---

## 10. Interaction & UX patterns

**Navigation hierarchy:**
1. Flow rail = primary stage switcher
2. View toolbar = workbench vs graph + optional panels
3. Stage headers = stage-specific primary actions

**Real-time behavior:**
- WebSocket streams live events into Sources stage preview
- Source Start/Stop controls ingestion
- Throughput metric derives from running source count
- System log appends timestamped operational messages

**Progressive disclosure:**
- Metrics and Log hidden by default (reduce clutter)
- Export adapter fields change based on selected adapter
- External source fields only show in External mode
- Advanced export JSON collapsed by default

**Disabled states:**
- Test processing requires selected model
- Backfill requires connected lake
- Federated round actions require active connection
- Many actions require selected pipeline

**Empty states:**
- No pipeline → centered CTA
- No sources / no events / no export targets / no federated rounds → dashed bordered placeholders

**Responsive:**
- Sidebar collapses on mobile
- Flow rail: 4 → 2 → 1 columns
- Stage layouts stack from 2-column to single column
- Workflow graph horizontal scroll below ~900px

---

## 11. Information architecture summary (for wireframes)

```
[App Shell: Sidebar]
└── Pipeline Workbench (Web Page)
    ├── Header (pipeline picker + global actions)
    ├── Toolbar (Workbench | Workflow) + (Metrics | Log)
    ├── Flow Rail [Sources] [Core Unit] [Lake] [Federated]
    ├── Content Area (one visible)
    │   ├── Empty State
    │   ├── OR Workflow Graph (3 columns)
    │   └── OR Workbench Stage Panel
    │       ├── Sources (list + events)
    │       ├── Processing (model + guards summary)
    │       ├── Lake & Export (browser + backfill + export form)
    │       └── Federated (coordinator + rounds + training)
    ├── [Optional] Metrics Panel
    ├── [Optional] System Log
    └── Modals: Create Pipeline | Connect Source | Connect Lake | Core Unit
```

---

## 12. Designer notes

- Treat this as an **operator console**, not a marketing page — dense but structured
- Emphasize the **4-stage flow rail** as the visual spine
- **Core Unit modal** is the control center for model + preprocessing guards — should feel like “engine configuration”
- Differentiate **live preprocessing model** (Core Unit) from **federated training** (separate stage)
- Use **teal** for primary actions and active navigation; **monospace** for technical values (endpoints, model IDs, logs)
- Modals must be **viewport-centered** with full-screen dim overlay (critical UX requirement)
- Dark theme with subtle grid/gradient atmosphere; cards float on layered dark panels
