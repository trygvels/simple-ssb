
# SSB Agent Improvement Guide

This guide turns the previous assessment into focused, *general* implementation guidance for a coding agent that improves an SSB PxWeb v2–based data analysis agent. It emphasizes **fewer, more useful tool calls**, **faster relevant data retrieval**, and **clearer answers** — without hardcoded paths or table‑specific tricks.

> **Non-goals**
> - No municipality→county fallbacks for longer time series.
> - No hardcoded synonym lists or domain-specific query rewrites.
> - Do **not** force a fixed number of tool calls. Prefer the *minimal* set that answers the question, but allow early exits if sufficient info is already known.
> - `discover_dimension_values` is **optional**; skip it when you clearly know codes or can resolve them from metadata.

---

## 1) What we observed (root causes of “overwork”)

- **Redundant searches**: Multiple `search_tables` calls for the same concept.
- **Dimension search misuse**: A single `search_term` containing multiple names (e.g., `"Oslo Porsgrunn"`) yielded no matches; codes were then guessed.
- **Unneeded retrieval**: Fetching both `ArbBostedRegion = 1` and `2` when only one is used in the final answer.
- **Telemetry mismatch**: “Retrieved 20/14 points” confusion due to how totals/returned are counted.
- **Missing logs**: Success path in `get_filtered_data` returns before logging.
- **High HTTP overhead**: New `httpx.AsyncClient` per request; no pooling/caching.
- **Broad dimensions**: `Region` has ~910 values; no narrowing when searching label→code.

---

## 2) General workflow (minimal, flexible)

The agent should prefer a short path, but **may skip steps** when it already knows enough (e.g., table id + filters).

**Typical minimal path (not mandatory):**
1. `search_tables` — **once** — using a simple concept-oriented query derived from user intent.
2. `get_table_info(include_structure=True)` — **once** — confirm dimensions, time span, and read category metadata + codeLists.
3. `get_filtered_data` — **once** — fetch all needed regions/categories in one go.

**The agent may skip:**
- `search_tables` if it already knows a **table id** that matches the intent.
- `discover_dimension_values` if it can resolve codes from **metadata** (`get_table_info`) or the user supplied codes.
- Extra `get_filtered_data` calls for alternate slices **unless** explicitly requested or needed for the answer.

**Use `discover_dimension_values` only when:**
- You’re **not** receiving results and suspect a label/code mismatch.
- You’re **unsure** about allowed values for a dimension (e.g., what’s valid for `MaaleMetode` or a non-obvious categorical).
- You need **fuzzy matching** across many labels and metadata resolution was inconclusive.

---

## 3) Concrete, general code changes

### 3.1 Fix `get_filtered_data` logging and counters
- Always build a `result` dict, **log it**, then return.
- Report consistent counters: `observations_total` (raw vector length or clearly defined rule) and `observations_returned` (decoded rows).

```python
# After decoding to `recs`, computing `dims` and optional `summary`:
observations_total = len(data.get("value", []) )  # choose a clear, consistent definition

result = {
    "table_id": table_id,
    "title": data.get("label", f"Table {table_id}"),
    "source": data.get("source", "Statistics Norway"),
    "updated": data.get("updated", ""),
    "filters_applied": filters,
    "dimensions": list(dims.keys()),
    "observations_total": observations_total,
    "observations_returned": len(recs),
    "formatted_data": recs,
    **({"summary_stats": summary} if summary else {})
}

log_tool_call(LOG_FILE, "get_filtered_data", args, result)
return result
```

Console summary:
```python
elif tool_name == "get_filtered_data":
    tot = parsed.get("observations_total", 0)
    ret = parsed.get("observations_returned", 0)
    console.print(f"   [green]✓ Retrieved {ret}/{tot} observations[/green]")
```

### 3.2 Multi-term label search + optional codeList narrowing (generic for any dimension)
Allow `discover_dimension_values` to accept **multiple terms** (comma/whitespace-separated) and an **optional `code_list`** hint to narrow the candidate set (e.g., municipalities vs counties). Keep it generic for any dimension that has large cardinality.

```python
@function_tool
async def discover_dimension_values(table_id: str, dimension_name: str,
                                    search_term: str = "", include_code_lists: bool = True,
                                    code_list: str = "") -> dict:
    meta_url = f"https://data.ssb.no/api/pxwebapi/v2-beta/tables/{table_id}/metadata"
    meta = await robust_api_call(meta_url, {"lang": "no"})
    dim = meta["dimension"][dimension_name]
    cat = dim.get("category", {})
    codes_all = _codes_in_order(cat.get("index"))
    labels = cat.get("label", {}) or {}

    # Optional narrowing heuristic (generic): filter candidate codes based on format or a known subset.
    codes = codes_all
    if code_list:
        # This is a generic place to narrow; avoid extra API calls.
        # Example heuristic: keep 4-digit numeric codes for municipality-like sets,
        # or use a predicate you define for the given dimension.
        codes = [c for c in codes_all if c.isdigit() and len(c) in (2, 4)]  # generic numeric narrowing

    terms = [t.strip().lower() for t in re.split(r"[,;\s]+", search_term) if t.strip()] if search_term else []
    def matches(c):
        lab = labels.get(c, c)
        s = f"{c} {lab}".lower()
        return not terms or any(t in s for t in terms)

    pairs = [(c, labels.get(c, c)) for c in codes if matches(c)]
    # ...build response as today...
```

> Use this for **any** dimension with many labels. It’s a *hinted* narrowing, not a hard rule.

### 3.3 Resolve codes from **metadata** (avoid extra calls)
If `get_table_info(include_structure=True)` already brought you `dimension[<name>].category`, you can resolve label→code **without** calling `discover_dimension_values`. Keep this generic: it works for any dimension with label maps.

```python
def resolve_codes_from_metadata(meta: dict, dim_name: str, labels_or_codes: list[str]) -> list[str]:
    dim = meta.get("dimension", {}).get(dim_name, {})
    cat = dim.get("category", {})
    codes = _codes_in_order(cat.get("index"))
    label_map = { (cat.get("label") or {}).get(c, c).lower(): c for c in codes }
    out = []
    for token in labels_or_codes:
        t = token.strip()
        if t in codes:
            out.append(t)
        else:
            key = t.lower()
            # exact, then prefix/substring fallback
            if key in label_map:
                out.append(label_map[key])
            else:
                # prefix/substring scan
                for lab_lower, code in label_map.items():
                    if lab_lower.startswith(key) or key in lab_lower:
                        out.append(code)
                        break
    return out
```

**Policy:** Try this first. If it yields an empty/ambiguous set, *then* call `discover_dimension_values` with `search_term` to probe allowed values.

### 3.4 GET vs POST heuristic (uniform behavior across dimensions)
Prefer POST whenever a selector is a list, contains a comma, or uses a pattern (`top(...)`, `from(...)`, `[range(...)]`, `*`, `?`).

```python
def _needs_post(filters: dict) -> bool:
    for sel in filters.values():
        if isinstance(sel, list):
            return True
        s = str(sel)
        if any(tok in s for tok in [",", "(", "[", "*", "?"]):
            return True
    return False
```

Use:
```python
use_post = _needs_post(filters) or len(url_with_params) > 2000
```

### 3.5 Reuse a single HTTP client + add metadata cache
- Create one `httpx.AsyncClient`; reuse for all calls.
- Add a small LRU for `/metadata` responses keyed by `(table_id, lang)`.

```python
_http_client: httpx.AsyncClient | None = None
_meta_cache: dict[tuple[str,str], dict] = {}

async def _get_client():
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT, limits=_HTTP_LIMITS, headers=_DEFAULT_HEADERS)
    return _http_client

async def robust_api_call(...):
    client = await _get_client()
    # ... existing retry logic using `client` ...

async def get_table_info(...):
    # fill _meta_cache[(table_id, lang)] = meta
```

### 3.6 Minimize calls by **skipping** what you already know
- **Skip** `search_tables` if the agent knows a **table id** matching the intent.
- **Skip** `discover_dimension_values` if codes can be resolved from metadata or are provided by the user.
- **Combine** all needed categories (e.g., multiple regions) in a **single** `get_filtered_data` call.

### 3.7 Only fetch what you’ll use (generic defaults, not hardcoded)
- Infer sensible defaults from user intent (e.g., aggregate categories like “both genders”, “all ages”), but **don’t enforce** them: if the prompt specifies otherwise, override.
- Example generic pattern across dimensions:
  - If a dimension has a value named like “I alt”/“Total” → prefer that unless the question narrows it.
  - If a “location scope” dimension exists (e.g., residence vs workplace) → pick one based on the wording, not both.

> Keep this logic **generic**. Do not freeze specific codes/labels in code; read them from metadata and pick by label matching (“total”, “all”, etc.).

### 3.8 Cleaner console UX (optional but helpful)
- Show at most **one** search result (the chosen table) to reduce noise.
- Print the **final numbers first**, then period coverage, then any data availability caveat.
- Keep the **source** line explicit (table id + title).

---

## 4) Decision rules the agent can follow (concise)

- **If table id + filters are known** → call `get_filtered_data` directly.
- **Else** `search_tables` once with a concept-oriented query; pick the first clearly relevant table.
- **Always** call `get_table_info(include_structure=True)` on the chosen table.
- **Resolve codes** from metadata first; only **then** consider `discover_dimension_values` when unsure or on failure.
- **Prefer one** `get_filtered_data` with all needed values in the same dimension (e.g., multiple regions) unless the task requires multiple slices.
- **Use POST** when selectors are lists/patterns or the URL would be long.
- **Log** every successful call; keep counters consistent and meaningful.

---

## 5) Lightweight examples (pattern, not prescription)

- **Known table id case**: user asks monthly unemployment rate by age group in 2022; the agent knows the table id and labels → go straight to `get_filtered_data` with a POST request expanding `Tid=[range(2022,2022)]` and `Alder=[...]` values.
- **Unknown label case**: user mentions a rare category name; metadata resolution fails → call `discover_dimension_values(dimension="Category", search_term="<name>")` and then retry data fetch with the returned code.

> These are illustrative patterns only — keep the logic **general** and data‑driven via metadata.

---

## 6) Checklist (for PRs / reviews)

- [ ] Only **one** `search_tables` unless intent changes mid-run — or skip if id known.
- [ ] `get_table_info(include_structure=True)` once; metadata cached.
- [ ] Try metadata-based code resolution before calling `discover_dimension_values`.
- [ ] `discover_dimension_values` supports **multi-term** search and optional **code_list** narrowing.
- [ ] `get_filtered_data` called **once** per answerable slice; combine multiple values in one call.
- [ ] POST heuristic used for lists/patterns/long URLs.
- [ ] Success path is **logged**; counters are consistent.
- [ ] Single reusable HTTP client; metadata LRU cache in place.
- [ ] Console output: numbers → coverage → caveats → source.

---

## 7) Appendix: small helpers

```python
def resolve_codes_from_metadata(meta: dict, dim_name: str, labels_or_codes: list[str]) -> list[str]:
    dim = meta.get("dimension", {}).get(dim_name, {})
    cat = dim.get("category", {})
    codes = _codes_in_order(cat.get("index"))
    labels = (cat.get("label") or {})
    label_to_code = { labels.get(c, c).lower(): c for c in codes }
    out = []
    for token in labels_or_codes:
        t = token.strip()
        if t in codes:
            out.append(t)
        else:
            key = t.lower()
            if key in label_to_code:
                out.append(label_to_code[key])
            else:
                # loose match
                for lab_lower, code in label_to_code.items():
                    if lab_lower.startswith(key) or key in lab_lower:
                        out.append(code); break
    return out

def _needs_post(filters: dict) -> bool:
    for sel in filters.values():
        if isinstance(sel, list):
            return True
        s = str(sel)
        if any(tok in s for tok in [",", "(", "[", "*", "?"]):
            return True
    return False
```

---

### Final note
This guide is intentionally **general**: it avoids table-specific shortcuts, fallbacks, and hardcoded paths. It leans on table **metadata**, minimal tool calls, and explicit decision points to keep the agent fast, reliable, and extensible across domains.
