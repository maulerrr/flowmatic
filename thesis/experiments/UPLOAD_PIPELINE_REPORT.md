# Upload Pipeline Experiment Report

- **Completed at**: `2026-05-25T17:07:59.707716+00:00`
- **API**: `http://127.0.0.1:8080/api/v1`

## Notes

- Successful ingestion upload returned **HTTP 201 Created**; treat any **2xx** as success for POST /ingestion/upload.



1. Upload validates CSV, stores source, queues pipeline job.
2. Worker parses CSV, runs quality + cleaning, stores metrics and summary on the run.

## Outcome snapshot

| Step | OK | Detail |
| --- | --- | --- |
| Session | yes | register 400 login 200 |
| Upload | yes | HTTP 201 in 0.225s |
| Pipeline | `completed` | polled 1.885s |

## Metrics

- **status**: completed
- **rowsIngested**: 30000
- **rowsCleaned**: 30000
- **rowsErrors**: 0
- **processingTimeMs**: 406
- **resultFile**: `astana_synthetic_data.csv_cleaned` id `cmplgmf5u0006p93cn2mz08a7` size `2182436`

### Stored pipeline summary (snippet)

```json
{"overview":"Analysis of the 30,000 records from astana_synthetic_data.csv. The dataset is highly clean, with 30,000 records successfully processed. ","scores":{"initial":100,"final":100},"insights":["Identified 5 numeric fields and 4 categorical fields.","No significant schema violations or anomalies were detected."],"recommendation":"Proceed with downstream analytics."}
```

### Preview stats

```json
{
  "rowsIngested": 30000,
  "rowsCleaned": 30000,
  "rowsErrors": 0,
  "processingTimeMs": 406
}
```

### Parsed summary.scores

```json
{
  "initial": 100,
  "final": 100
}
```

## Export

- History endpoint HTTP `200` (explicit POST export required).

```json
{
  "success": true,
  "data": []
}
```

## Timing (seconds)

```json
{
  "auth": 0.112,
  "upload": 0.225,
  "polling": 1.885,
  "total": 2.3
}
```
