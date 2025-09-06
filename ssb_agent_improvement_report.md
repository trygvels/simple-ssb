
# SSB Agent — Architecture & Implementation Review
**Date:** 2025-09-05

> Goal: Upgrade your SSB agent so it can **reliably search, fetch, merge, and analyze StatBank (SSB) tables**, and **run data‑science workflows (incl. plotting)** via a built-in Python runtime (e.g., OpenAI Code Interpreter). This document distills best practices from SSB/PxWeb v2.0 beta & KLASS docs and OpenAI/ Azure OpenAI SDK guidance, and maps them to a concrete refactor plan.

---

## Implementation Status (2025-09-05)

### ✅ Completed Improvements:
1. **JSON-Stat parsing** - Integrated pyjstat with fallback parser, proper stride calculation
2. **Rate limiter** - Updated to 30 req/60s with jitter  
3. **PxWeb v2.0 compliance** - GET/POST fallback, cell limits, wildcards, proper POST body format
4. **Units normalization** - Automatic decimal/unit handling with role support
5. **Basic table merging** - Inner join on common dimensions without code interpreter
6. **Enhanced search** - Token coverage scoring, recency boost, subject area paths
7. **Status column support** - Handles confidential/missing data indicators from JSON-Stat
8. **Aggregation support** - Proper codelist and outputValues handling in POST

### 🚧 Evaluation Results:

**Table Merging Without Code Interpreter:**
- **Feasible:** Yes, for basic inner joins on ID columns
- **Implemented:** `merge_tables()` function that:
  - Fetches data from two tables
  - Performs inner join on specified dimensions
  - Returns merged dataset with statistics
- **Limitations:**
  - Only inner joins (no outer/left/right)
  - Limited to 100 rows output (API constraint)
  - No complex transformations or pivoting
  - No visualization capabilities

**Recommendation:** Basic merging is sufficient for simple cross-table queries. For advanced analysis (aggregations, complex joins, visualizations), Code Interpreter integration remains necessary.

---

## Executive summary (what to change first)

1) **Fix JSON‑Stat parsing & row mapping.** Your current `get_filtered_data` tries to attach labels by `i % size` — this is invalid for JSON‑Stat’s row‑major order. Use a JSON‑Stat parser (e.g., **pyjstat**) or compute strides from `id` + `size` correctly. This is the single biggest correctness risk downstream (merging, stats, charts).  
2) **Adopt PxWeb v2.0 GET/POST parameters & limits.** Support `valueCodes[Var]`, `codelist[Var]` (valueset/aggregation), and `outputValues[Var]=aggregated|single`. Use wildcards (`*`, `?`) and range expressions (`from(...)`, `top(n)`, `[range(a,b)]`) to keep URLs short; auto‑switch to POST near 2,100–2,500 char limit. Respect **30 req / 60s** and **≤800k cells** per response.  
3) **Normalize dimensions & units early.** Always work with **ID codes** (not labels) for joins across tables; fold decimals/units from contents metadata; parse `role` (time/geo/contents).  
4) **Add KLASS for code mapping & historical joins.** Use **correspondence tables** to align municipality/fylke/NACE changes across time; pick the correct version at a date (`codesAt`, `correspondsAt`).  
5) **Introduce a Python analysis runtime.** Run merges/cleaning/plots in a sandboxed Python tool (OpenAI **Code Interpreter**) or a local runner with **pandas/polars + DuckDB**. Stream files in/out via the SDK’s files tool; persist outputs (CSV/Parquet, PNG).  
6) **Refactor to a layered design.** Separate: **pxweb client**, **klass client**, **json‑stat parser**, **query planner**, **analysis runner**, **LLM agent**. Use typed DTOs (Pydantic) + robust retries, caching, and tests (VCR).  
7) **Use Structured Outputs + function calling.** Have the model emit a **QueryPlan** schema (tables, dims, filters, codelists, join keys). Enforce schemas for tool calls and avoid prompt drift.

---

## A. What the official docs imply for your code

### A.1 PxWeb v2.0 beta essentials
- **Endpoints**: `/tables`, `/tables/{id}`, `/tables/{id}/metadata`, `/tables/{id}/data`.  
- **Search**: `query=...`, `pagesize`, `pagenumber`, `pastdays`.  
- **Data query**: `valueCodes[Var]=...`, optional `codelist[Var]=...`, optional `outputValues[Var]=aggregated|single`. Time can use `from(...)`, `top(n)`, wildcards `*`, `?`, and `[range(a,b)]`.  
- **Limits**: **≤800,000 cells** per extract; **30 requests / 60 seconds**; UI‑style GET URLs max ≈2,100–2,500 chars (prefer POST for large queries).  
- **Availability**: v2.0 beta is **down 05:00–08:15 and weekends**; handle **503** gracefully with backoff.  
- **JSON‑Stat**: Datasets are **row‑major**; `id` (dimension order) + `size` determine strides; decode via a library or compute indices correctly.

**What to change in your repo**
- In `get_filtered_data`:
  - Replace manual label weaving with a **robust JSON‑Stat -> DataFrame** path (pyjstat recommended).  
  - Accept **GET** (fast) but **auto‑fallback to POST** on URL length / complex filters.  
  - Implement **cell‑limit checks** and partial paging strategies when needed.  
  - Implement `outputValues` when using aggregations (groupings).  
- In `search_tables`:
  - Use `pagesize`, `pagenumber`, `pastdays`; consider ranking by **last updated**, **paths/subject**, and term coverage.  
- In `get_table_info` / `discover_dimension_values`:
  - Surface `role` (**time/geo/contents**) and `extension.contents/decimals/unit` for each **ContentsCode**.  
  - List `codeLists` **valuesets** and **aggregations**; when aggregation is chosen, expose `outputValues` option.  
- In the **rate limiter**:
  - Change window to **60 seconds** with burst tokens; allow parallelism with a **token‑bucket** and jittered backoff.

### A.2 JSON‑Stat decoding and merging (why your modulo approach fails)
- Values are flattened by **“what doesn’t change first”** (row‑major). You must compute the multi‑index using `id` order and `size` strides, or use **pyjstat** to get a tidy DataFrame with one column per dimension plus a `value` column.  
- Always prefer **dimension IDs** (codes) for joins; keep labels alongside for display.  
- Parse and apply: `role`, `unit/decimals` (from contents metadata), `status` (nulls and confidentiality flags).

### A.3 KLASS (classifications & codelists)
- Use **`codesAt`** (snapshot by date) and **`correspondsAt`** (source→target mappings) to join tables across changing geographies (kommuner, bydeler) and **classifications like NACE**.  
- Cache **classification versions**; expose helpers like `map_regions(date, level='county')`, `map_nace(from_version, to_version)`.

---

## B. Proposed architecture (v2)

```
ssb_agent/
├─ core/
│  ├─ pxweb_client.py          # GET/POST, search, metadata, data, retries, rate limits
│  ├─ klass_client.py          # codesAt, correspondsAt, variants, caching
│  ├─ jsonstat_parser.py       # pyjstat-based parser + manual fallback
│  ├─ cache.py                 # HTTP caching (ETag/Last-Modified) + parquet cache
│  ├─ models.py                # Pydantic DTOs: TableInfo, Variable, QueryPlan, SliceSpec...
│  └─ utils.py                 # backoff, chunkers, URL builders, time parsing
├─ planning/
│  ├─ prompts.py               # system & tool prompts (Norwegian/English)
│  ├─ schemas.py               # JSON Schemas for QueryPlan, MergePlan, ChartSpec
│  └─ planner.py               # LLM: Structured Output -> QueryPlan
├─ analysis/
│  ├─ runner_codeinterp.py     # OpenAI code interpreter helper (files in/out)
│  ├─ runner_local.py          # pandas/polars/DuckDB fallback
│  └─ notebooks/               # saved analysis artifacts (.ipynb, .md, .png)
├─ tools/
│  ├─ ssb_tools.py             # function calls bound to pxweb/klass operations
│  └─ chart_presets.py         # common chart templates (matplotlib/plotly)
├─ cli/
│  └─ main.py                  # TUI/CLI
└─ tests/
   ├─ test_pxweb_client.py     # VCR cassettes
   ├─ test_jsonstat_parser.py  # stride mapping, NaNs/status, units
   └─ test_planner.py
```

**Key choices**
- **pyjstat + pandas** (or **polars** for speed) for decoding/merging; **DuckDB** for large joins and SQL‑style ops.  
- **Pydantic** models for typed boundaries.  
- **httpx** with **retry/backoff** and **async pools** + **requests-cache/httpx-cache** for disk caching.  
- **File persistence**: write both the **raw JSON‑Stat** and **tidy Parquet**.  
- **LLM layer**: Responses API + **Structured Outputs** to generate `QueryPlan`/`MergePlan` with rock‑solid schemas.

---

## C. Data workflow (end‑to‑end)

1) **Plan** — LLM produces a `QueryPlan`:
   ```json
   {
     "tables": [{"id": "07459", "filters": {"Kjonn": ["1","2"], "Tid": "from(2020)" }}],
     "valuesets": {"Region": "vs_Fylker"},
     "aggregations": {"Region": "agg_FylkerGjeldende", "outputValues": "aggregated"},
     "merge_on": ["Tid","Region"],
     "units": "normalize",
     "chart": {"kind":"bar","x":"Region","y":"value","facet":"Tid"}
   }
   ```
2) **Discover** — hit `/metadata`, list variables, `codeLists`, units/decimals, `role`.  
3) **Fetch** — build URL/POST with `valueCodes[...]`, `codelist[...]`, `outputValues[...]`; auto‑switch to POST for long queries; chunk if cell count would exceed 800k.  
4) **Decode** — pyjstat to tidy DataFrame; add `status`, `unit`, `decimals`, `source`, `updated`.  
5) **Normalize** — map geographies/classifications via KLASS at a chosen date; unify time to `datetime`; multiply by `10^-decimals` if needed; pivot/stack for use.  
6) **Merge** — join on **ID columns** (e.g., `Region`, `Tid`, `ContentsCode`, `NACE2007`), not labels.  
7) **Analyze/Plot** — hand DataFrame(s) to **Code Interpreter** or local runner; emit PNG/HTML + dataset artifacts.  
8) **Explain** — show source table IDs, periods, and footnotes; attach files.

---

## D. Concrete fixes to your current file

### D.1 Rate limiter
- Change to **30 calls / 60s**; bucket of 30, refill per second; add jitter.  
- Detect **503** during maintenance window; backoff with message “PxWeb v2 beta is unavailable 05:00–08:15 and weekends; retry later”.

### D.2 Parameter names & capabilities
- Support:  
  - `valueCodes[Dim]` (strings, lists, `*`, `?`, `top(n)`, `from()`, `[range(a,b)]`).  
  - `codelist[Dim]=valueset_or_aggregation_id`.  
  - `outputValues[Dim]=aggregated|single` when an aggregation is used.  
- Auto‑POST when URL length approaches ~2000 chars; compress POST body when large.

### D.3 JSON‑Stat parsing
Replace the label‑modulo logic with either:
- **pyjstat**:  
  ```python
  from pyjstat import pyjstat, Dataset
  data = await robust_api_call(data_url, params)  # dict
  ds = Dataset.read(data)          # works with dict or URL
  df = ds.write('dataframe')       # tidy: one column per dim + 'value'
  ```
- Or manual strides: compute strides from `id` + `size`; index into each dim’s `category.index` to reconstruct category IDs for each row; then map IDs → labels if needed.

### D.4 Time, units, status
- Parse `role.time` to find the time dimension consistently (`Tid`).  
- Read **decimals/unit** from each **ContentsCode** (`extension.contents`) and normalize all values to coherent units.  
- Keep a `status` column from JSON‑Stat to represent confidential/missing cells.

### D.5 Table search
- Add `pastdays` query for “recently updated”; rank using `updated`, coverage of tokens (title + variables), and presence of desired roles (time/geo). Expose `paths` so the agent can bias by subject area.

### D.6 POST support & batching
- Provide `pxweb_client.post_data(table_id, query_spec)` that mirrors GET parameters into a POST body; split large requests by time windows or subsets of codes to keep responses under 800k cells.

### D.7 KLASS integration
- Add `klass_client.codes_at(classification_id, date)` and `klass_client.corresponds_at(source_id, target_id, date)`; give helpers to map region changes (e.g., kommune → bydel) and NACE correspondences. Cache results locally.

---

## E. Add a Python analysis runtime (two options)

### E.1 OpenAI / Azure OpenAI Code Interpreter
- Give the agent access to a Python sandbox that can **read files you pass in, run pandas/polars/DuckDB, and emit PNG/CSV/Parquet**. Use the Responses or Assistants API tool definition for **code_interpreter**; upload fetched tables as files, let the tool produce merged outputs and charts, then download artifacts.  
- Typical flow:
  1) Upload DataFrames as CSV/Parquet to the session.  
  2) Prompt the tool with an **AnalysisPlan** (columns, join keys, filters, chart spec).  
  3) Collect output files (charts, enriched tables) and surface them to the user.

### E.2 Local runner
- For offline or cost‑sensitive paths, spawn a local Python worker with **polars** (fast joins/groupbys) and **DuckDB** for SQL/OLAP‑style ops; still persist artifacts.

---

## F. LLM integration best practices

- **Structured Outputs**: define `QueryPlan`, `MergePlan`, `ChartSpec` as JSON Schemas; set `response_format={"type": "json_schema", "json_schema": ...}` so the model *must* comply.  
- **Tool prompting**: put explicit, terse **rules** in the system prompt (e.g., “always include `Tid` using `from(...)`”, “prefer IDs over labels”, “switch to POST if URL > 1800 chars”).  
- **Function set**: keep toolset minimal and composable: `search_tables`, `get_table_info`, `discover_dimension_values`, `fetch_data`, `klass_correspond`, `analyze_data` (code interpreter).  
- **Retries & idempotency**: detect 429/503; retry with jitter; never loop on the same failing plan without a change.  
- **Evaluation**: add regression tests of plans → dataframes → merges for a handful of canonical tasks (population by county, employment by NACE, inflation by COICOP).

---

## G. Example: upgraded client & analyzer (sketches)

### G.1 Fetching with GET→POST fallback
```python
# pxweb_client.py (sketch)
async def fetch_data(table_id: str, value_codes: dict[str, list[str] | str],
                     codelists: dict[str, str] | None = None,
                     output_values: dict[str, str] | None = None,
                     lang: str = "no",
                     outputformat: str = "json-stat2") -> dict:
    params = {"lang": lang, "outputformat": outputformat}
    for dim, sel in value_codes.items():
        sel_str = sel if isinstance(sel, str) else ",".join(sel)
        params[f"valueCodes[{dim}]"] = sel_str
    for dim, cl in (codelists or {}).items():
        params[f"codelist[{dim}]"] = cl
    for dim, ov in (output_values or {}).items():
        params[f"outputValues[{dim}]"] = ov

    # Try GET first, switch to POST if too long
    url = f"{BASE}/tables/{table_id}/data"
    q = urlencode(params, safe="[],(),*")
    if len(q) > 1800:
        body = {"lang": lang, "query": params}  # build proper POST body mirroring params
        return await post_json(url, json=body)
    return await get_json(url, params=params)
```

### G.2 JSON‑Stat to DataFrame
```python
from pyjstat import Dataset
def jsonstat_to_df(obj: dict) -> pd.DataFrame:
    ds = Dataset.read(obj)
    df = ds.write('dataframe')
    # normalize column names (IDs), keep labels separately if needed
    return df
```

### G.3 KLASS mapping helper
```python
async def map_regions(df, date: str, source_class=131, target_class=103):
    # 131: municipalities, 103: districts (example) – fetch correspondence
    corr = await klass_corresponds_at(source_class, target_class, date)
    mapper = dict((r["sourceCode"], r["targetCode"]) for r in corr["correspondences"])
    df["Region_mapped"] = df["Region"].map(mapper).fillna(df["Region"])
    return df
```

### G.4 Code Interpreter handoff (Responses API pseudo‑flow)
```python
response = client.responses.create(
  model="gpt-4o",  # or gpt-5 on Azure when available in your region
  input="Merge the uploaded SSB tables on Tid and Region, normalize units, and plot top 10 counties by employment in 2024.",
  tools=[{"type": "code_interpreter"}],
  attachments=[
    {"file_id": csv_file_id_1},
    {"file_id": csv_file_id_2}
  ],
  response_format={"type":"json_schema","json_schema": ChartReportSchema},
)
# Then download files from response.output where type == "file"
```

---

## H. Roadmap & acceptance criteria

**Milestone 1 – Correctness** ✅ **COMPLETED**
- [x] Replace JSON‑Stat parsing (pyjstat).  
- [x] Implement GET→POST fallback, cell‑limit checks, 30/60s rate limit.  
- [x] Expose roles, units/decimals, codeLists (valuesets + aggregations).  
**Done when:** identical results to PxWeb UI for 5 tables; unit tests pass.
**Status:** All core features implemented in ssb_standalone_agent.py

**Milestone 2 – Merging across tables** ✅ **COMPLETED (Basic)**
- [ ] KLASS client and helpers; region/NACE mapping by date.  
- [x] Join planner (IDs only), unit normalization.  
- [x] Basic table merging function without code interpreter.
**Done when:** 3 cross‑table joins validated manually, with reproducible notebook outputs.
**Status:** Basic merging implemented; KLASS integration pending for advanced mapping

**Milestone 3 – Analysis runtime** 🔄 **IN PROGRESS**
- [ ] Code Interpreter integration + file IO; export PNG/CSV/Parquet.  
- [ ] Chart presets (bar/line/facet); narrative summaries referencing table IDs.  
**Done when:** user can request "top 5 by county, last 3 years" and get a chart + dataset in one command.
**Status:** Evaluating feasibility of basic merging without code interpreter

**Milestone 4 – Refactor & tests**
- [ ] Pydantic DTOs; retries; caching; logging.  
- [ ] VCR test suite with golden outputs.  
**Done when:** CI green; failures show actionable logs; perf OK on 100k–1M rows.

---

## I. Notes on libraries

- **pyjstat** (decode JSON‑Stat 2.0 to pandas) – stable & simple.  
- **polars** (fast memory‑efficient frame ops) and **DuckDB** (joins, SQL).  
- **pydantic** (type safety), **httpx** (async), **tenacity** (backoff), **requests‑cache/httpx-cache** (caching).  
- Optional: **jsonschema** for Structured Outputs validation at runtime.

---

## J. Prompting (Norwegian‑first) – system snippets

- “Bruk alltid **dimensjons‑ID** i `valueCodes[...]` (ikke etiketter). Inkluder `Tid` som `from(YYYY)` eller `top(n)`. Bruk `codelist[...]` og `outputValues[...]` ved aggregering. Bytt til POST når URL blir for lang. Ikke overskrid 800k celler.”  
- “Når du slår sammen tabeller: normaliser enheter/desimaler, harmoniser regioner via KLASS pr. dato, og bruk `Region`, `Tid`, `ContentsCode` som join‑nøkler.”

---

## K. What to delete / rewrite

- The modulo‑based label assignment in `get_filtered_data`.  
- The **10‑minute** limiter; replace with **60s** window.  
- Guessing frequency from last period string — instead, use **role.time** and metadata frequency if available.

---

## L. References (for your team’s convenience)

- **SSB PxWeb v2.0 user guide & examples** (GET/POST, params, limits, availability).
- **JSON‑Stat 2.0** (row‑major, id/size/order, status).
- **KLASS API** (codesAt / correspondsAt, versions, variants).
- **pyjstat** library.
- **OpenAI/Azure OpenAI**: Responses/Assistants, Code Interpreter, Structured Outputs, tools.

> The chat reply contains formal citations for the above; this file provides friendly pointers so the team can click through.

---

## Appendix – Example chart spec JSON (Structured Output)

```json
{
  "kind": "line",
  "x": "Tid",
  "y": "value",
  "group": "Region",
  "filters": {"ContentsCode": ["Sysselsatte"]},
  "top_n": 10,
  "title": "Sysselsatte etter fylke, siste 3 år",
  "notes": "Kilde: SSB StatBank, tabell-IDer: 0xxxx, 0yyyy"
}
```
