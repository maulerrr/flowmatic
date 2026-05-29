# Flowmatic Thesis Reference Audit Report

**Thesis:** *Development of an Intelligent Assistant for Data Preparation Automation in Urban Transportation Management Systems*  
**PDF scanned:** `memoirthesis-flowmatic-v2.pdf`  
**Bibliography source:** `thesisbiblio.bib`  
**Audit date:** 2026-05-28  
**Rule applied:** entries listing **Ramazan Seiitbek / Ramazan Seiitbek Bakytuly / R. Seiitbek / Bakytuly, R.** were identified but not changed in this pass.

## Method

- Read the official Consensus MCP documentation at `https://docs.consensus.app/docs/mcp`.
- Used the already available Consensus MCP `search` and `fetch` tools. The docs say the official endpoint is `https://mcp.consensus.app/mcp`, the `/mcp` path is required, search returns peer-reviewed paper metadata, and `fetch` retrieves full details for a paper returned by search.
- Checked the compiled bibliography order from `memoirthesis.bbl` because PDF reference numbers are generated from that file.
- Used official/publisher pages as the authority where Consensus had no result or only indexed a preprint form.

## Problematic Or Risky References

| Current PDF ref | Key | Status | Evidence | Action |
|---:|---|---|---|---|
| 7 | `mahasivam2017dataprep` | Verified replacement for the old false/wrong data-preparation citation. | Consensus found and fetched *Data Preparation as a Service Based on Apache Spark*, Mahasivam/Nikolov/Sukhobok/Roman, 2017, pp. 125-139. | No change needed. The previous Procedia metadata problem is no longer present. |
| 15 | `bakytuly2024hybrid` | Verified in Consensus; author-listed Ramazan entry left untouched. | Consensus found and fetched the DSIT 2024 paper with Diar Begisbayev, A. Sakhipov, Ramazan Seiitbek, Aiganym Mansurova, Aigerim Mansurova, pp. 1-5. | No edit because this is a Ramazan-listed publication. |
| 16 | `yedilkhan2025flowmatic` | Not found in Consensus, but externally accessible via official KazATC/Crossref-linked metadata. Author-listed Ramazan entry left untouched. | Official KazATC PDF and ORCID/Crossref snippets confirm DOI `10.52167/1609-1817-2025-137-2-310-324` and authors including Ramazan Seiitbek. | No edit. Residual risk: not indexed by Consensus. |
| 17 | `bakytuly2025adaptive` | Verified in Consensus; author-listed Ramazan entry left untouched. | Consensus found and fetched the SIST 2025 paper with Aruzhan Mektepbayeva, Diar Begisbayev, Ramazan Seiitbek, N. Khaimuldin, A. Sakhipov, Daniyar Rakhimzhanov, pp. 1-8. | No edit because this is a Ramazan-listed publication. |
| 18 | `bakytuly2026microservices` | Not found in Consensus, but accessible on the official IJICT journal page. Author-listed Ramazan entry left untouched. | Official IJICT article page confirms title and DOI `10.54309/IJICT.2026.25.1.014`. | No edit. Residual risk: not indexed by Consensus; DOI indexing may lag. |
| 20 | `bakytuly2026architectural` | Unpublished / not externally verifiable. Author-listed Ramazan entry left untouched. | No Consensus result expected because the bibliography labels it as unpublished and not publicly available. | Keep only if Appendix B remains self-contained and the unpublished item is not used as sole evidence. |
| 40 | `benidis2022neural` | Metadata mismatch: current `.bib` year was 2022; official journal indexing gives ACM Computing Surveys 55(6), Article 121, 2023. | Consensus found/fetched the paper but reports the earlier 2020 preprint year. DBLP/DOI metadata identifies the ACM journal record as 2023, 55(6):121:1-121:36, DOI `10.1145/3533382`. | Corrected year to 2023 in `thesisbiblio.bib`. Key name retained to avoid citation churn. |
| 50 | `xu2021anomaly` | Verified; current bibliography uses the peer-reviewed ICLR 2022 version. | Consensus finds the arXiv version (2021). OpenReview confirms ICLR 2022 Spotlight with the same four authors. | No change needed. |

## Secondary Cleanup Findings

| Current PDF ref | Key | Finding |
|---:|---|---|
| 5 | `feurer2019auto` | Valid HPO chapter. The bibliography also includes the canonical auto-sklearn NeurIPS paper at ref 6. |
| 6 | `feurer2015autosklearn` | Consensus found/fetched the canonical auto-sklearn paper, 2015, pp. 2962-2970. |
| 56 | `event_driven2025` | DOI present in `.bib`; no immediate problem found from the current scan. |
| 57 | `kreps2011kafka` | Stable Microsoft Research URL present; acceptable as workshop/system reference. |
| 58 | `li2018diffusion` | OpenReview URL present; ICLR venue spelling is correct. |
| 62 | `ribeiro2016should` | DOI present; no immediate problem found. |

## Current Risk Summary

1. **Most serious remaining risk:** `bakytuly2026architectural` is unpublished and not externally verifiable. It is untouched because it lists Ramazan; keep the thesis appendix self-contained.
2. **Indexing gaps, not necessarily citation errors:** `yedilkhan2025flowmatic` and `bakytuly2026microservices` did not appear in Consensus, but official journal/Crossref-facing pages exist.
3. **Fixed in this pass:** `benidis2022neural` year corrected from 2022 to 2023 for the ACM Computing Surveys journal record.
4. **Already fixed before this pass:** the old false Procedia-style “Data preparation as a service” citation is no longer in the active bibliography; the current replacement is verified by Consensus.

## Files Updated

- `thesis/Flowmatic-Thesis-v2/thesisbiblio.bib`
- `thesis/Flowmatic-Thesis-v2/reference-audit-report.md`
