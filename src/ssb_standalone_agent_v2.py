#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSB Standalone Agent - Complete Norwegian Statistical Data Analysis Agent
(single file, no external parsing libs)

- Correct, pure-Python JSON-stat v2 decoding
- PxWebApi 2.0 beta param casing: valueCodes / codeList / outputValues
- Simple, safe GET→POST when URL length > ~2,000 chars
- Cell-limit estimation to respect 800k cells
- Join/merge on dimension codes (labels kept for display)
- No Code Interpreter, no pyjstat, no databases

Usage:
    python ssb_standalone_agent.py "Din spørring om norsk statistikk"

Example:
    python ssb_standalone_agent.py "Hvilken næring har flest sysselsatte? Gi meg top 5 i 2024"
"""

import asyncio
import os
import sys
import logging
import json
import time
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlencode
from pathlib import Path

import openai
import httpx
from dotenv import load_dotenv, find_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents import Agent, run, set_default_openai_client, set_default_openai_api, set_tracing_disabled
from agents import model_settings as agent_model_settings
from agents import function_tool

# =============================================================================
# Configuration
# =============================================================================

model = "gpt-5"  # your Azure deployment name is picked up via env if set
console = Console()

# Setup logging - concise
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=False, show_path=False, markup=True)]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
for handler in logging.root.handlers:
    if isinstance(handler, RichHandler):
        handler.setLevel(logging.WARNING)

# =============================================================================
# Helpers: Rate limiting, HTTP, JSON-Stat decoding, utilities
# =============================================================================

class RateLimiter:
    """Sliding-window limiter for SSB API (30 calls per 60 seconds)."""
    def __init__(self, max_calls: int = 30, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[datetime] = []

    async def acquire(self):
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)
        self.calls = [t for t in self.calls if t > cutoff]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]).total_seconds()
            if sleep_time > 0:
                # light jitter to avoid herd behavior
                await asyncio.sleep(min(sleep_time + min(1.0, 0.1 * sleep_time), 5.0))
        self.calls.append(datetime.now())

rate_limiter = RateLimiter()

# Metadata cache for reducing API calls
_meta_cache: Dict[Tuple[str, str], dict] = {}  # LRU cache for metadata

# Logging setup
def setup_logging():
    """Setup logging directory and return logger."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ssb_agent_{timestamp}.json"
    
    return log_file

def log_tool_call(log_file: Path, tool_name: str, args: dict, result: dict):
    """Log a tool call with its arguments and results."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "arguments": args,
        "result": result
    }
    
    # Append to log file
    with open(log_file, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')

# Global log file
LOG_FILE = setup_logging()

_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_HTTP_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)
_DEFAULT_HEADERS = {
    "User-Agent": "SSB-Standalone-Agent/1.0 (+https://github.com/your-org)",
    "Accept": "application/json",
}

async def robust_api_call(
    url: str,
    params: dict = None,
    json_body: dict = None,
    max_retries: int = 3
) -> Optional[dict]:
    """HTTP with retries, rate limiting, GET/POST support."""
    await rate_limiter.acquire()
    
    for attempt in range(max_retries):
        try:
            # Create a new client for each call to avoid connection issues
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT, limits=_HTTP_LIMITS, headers=_DEFAULT_HEADERS) as client:
                if json_body is not None:
                    resp = await client.post(url, params=params, json=json_body)
                else:
                    resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            if status == 503:
                return {
                    "error": "PxWebApi 2.0 beta is unavailable (maintenance window: 05:00–08:15 and weekends).",
                    "suggestion": "Prøv igjen senere i ukedager etter kl 08:15."
                }
            return {
                "error": f"HTTP {status} error from SSB API",
                "suggestion": "Kontroller API-parameterne og tabelltilgjengelighet."
            }
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                return {
                    "error": f"Connection failed after {max_retries} attempts: {str(e)}",
                    "suggestion": "Check network connection and SSB API availability."
                }
            await asyncio.sleep(2 ** attempt)
    return None

# ----------------------------- JSON-Stat Parsing ------------------------------

def _codes_in_order(category_index) -> List[str]:
    """Return codes in position order from category.index (dict or list)."""
    if isinstance(category_index, dict):
        # dict: {code: position}
        items = sorted(category_index.items(), key=lambda kv: kv[1])
        return [code for code, _pos in items]
    # list/array: already ordered
    return list(category_index or [])

def _make_dim_catalog(data: dict) -> Tuple[List[str], List[int], Dict[str, Dict[str, Any]]]:
    """Extract dimension order, sizes, and per-dim catalog (codes, labels)."""
    ids = data.get("id", [])
    sizes = data.get("size", [])
    dims = data.get("dimension", {})
    catalog: Dict[str, Dict[str, Any]] = {}
    for dim in ids:
        dim_obj = dims.get(dim, {})
        cat = dim_obj.get("category", {})
        codes = _codes_in_order(cat.get("index"))
        labels = cat.get("label", {}) or {}
        catalog[dim] = {"codes": codes, "labels": labels}
    return ids, sizes, catalog

def _compute_strides(sizes: List[int]) -> List[int]:
    """Compute row-major strides for JSON-stat value indexing."""
    n = len(sizes)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return strides

def _decode_record(i: int, ids: List[str], sizes: List[int], strides: List[int], catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Map a flat value index i → per-dimension code+label fields."""
    rec: Dict[str, Any] = {}
    for d_idx, dim in enumerate(ids):
        size = sizes[d_idx]
        stride = strides[d_idx]
        pos = (i // stride) % size
        codes = catalog[dim]["codes"]
        code = codes[pos] if pos < len(codes) else None
        label = catalog[dim]["labels"].get(code, code) if code is not None else None
        rec[f"{dim}_code"] = code
        rec[dim] = label
    return rec

def jsonstat_to_records(data: dict, max_points: int = 500) -> List[dict]:
    """
    Decode JSON-stat v2 to a list of records with dimension code+label and value.
    - Preserves 'status' strings if present (e.g., confidential/missing markers).
    - Does NOT rescale by 'decimals' (it's display precision, not a factor).
    """
    if not isinstance(data, dict) or "value" not in data or "dimension" not in data:
        return []
    ids, sizes, catalog = _make_dim_catalog(data)
    if not ids or not sizes:
        return []
    strides = _compute_strides(sizes)
    values = data.get("value", [])
    status_map = data.get("status", {}) or {}  # keys are string indices
    out: List[dict] = []
    for i, v in enumerate(values):
        if len(out) >= max_points:
            break
        # include rows with value or with explicit status
        s = status_map.get(str(i))
        if v is None and not s:
            continue
        rec = _decode_record(i, ids, sizes, strides, catalog)
        rec["value"] = v
        if s:
            rec["status"] = s
        out.append(rec)
    return out

# ----------------------------- Query utilities --------------------------------

def resolve_codes_from_metadata(meta: dict, dim_name: str, labels_or_codes: List[str]) -> List[str]:
    """Resolve label/code strings to codes using metadata - avoids extra API calls."""
    dim = meta.get("dimension", {}).get(dim_name, {})
    cat = dim.get("category", {})
    codes = _codes_in_order(cat.get("index"))
    labels = cat.get("label", {}) or {}
    label_to_code = {labels.get(c, c).lower(): c for c in codes}
    
    out = []
    for token in labels_or_codes:
        t = token.strip()
        # Direct code match
        if t in codes:
            out.append(t)
        else:
            key = t.lower()
            # Exact label match
            if key in label_to_code:
                out.append(label_to_code[key])
            else:
                # Prefix/substring fallback
                for lab_lower, code in label_to_code.items():
                    if lab_lower.startswith(key) or key in lab_lower:
                        out.append(code)
                        break
    return out

def _needs_post(filters: dict) -> bool:
    """Determine if POST is needed based on filter complexity."""
    for sel in filters.values():
        if isinstance(sel, list):
            return True
        s = str(sel)
        if any(tok in s for tok in [",", "(", "[", "*", "?"]):
            return True
    return False

_RANGE_RE = re.compile(r"^\[range\((.+?),(.+?)\)\]$", re.IGNORECASE)
_TOP_RE   = re.compile(r"^top\((\d+)(?:\s*,\s*(\d+))?\)$", re.IGNORECASE)
_FROM_RE  = re.compile(r"^from\((.+)\)$", re.IGNORECASE)

def _glob_to_regex(pat: str) -> re.Pattern:
    # Supports '*' and '?' anywhere (e.g., '2020*', '??')
    pat = re.escape(pat).replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(f"^{pat}$")

def select_codes_from_pattern(codes_in_order: List[str], selector: str, dim_name: str) -> List[str]:
    """Expand PxWeb pattern selectors to explicit code lists (for POST building)."""
    sel = selector.strip()
    if sel == "*":
        return list(codes_in_order)

    m = _TOP_RE.match(sel)
    if m:
        n = max(0, int(m.group(1)))
        if n <= 0:
            return []
        # For time 'Tid': take the last n (most recent). Otherwise: first n.
        if dim_name.lower() == "tid":
            return codes_in_order[-n:]
        return codes_in_order[:n]

    m = _FROM_RE.match(sel)
    if m:
        start = m.group(1)
        try:
            idx = codes_in_order.index(start)
            return codes_in_order[idx:]
        except ValueError:
            return []

    m = _RANGE_RE.match(sel)
    if m:
        a, b = m.group(1), m.group(2)
        try:
            ia = codes_in_order.index(a)
            ib = codes_in_order.index(b)
            lo, hi = sorted((ia, ib))
            return codes_in_order[lo:hi + 1]
        except ValueError:
            return []

    # Wildcards like '2020*' or '??'
    if "*" in sel or "?" in sel:
        rx = _glob_to_regex(sel)
        return [c for c in codes_in_order if rx.match(c)]

    # Single code
    return [sel]

def count_selected_for_dim(dim_meta: dict, selector) -> int:
    """Estimate how many values a selector picks in a dimension."""
    cat = dim_meta.get("category", {})
    codes = _codes_in_order(cat.get("index"))
    if isinstance(selector, list):
        return min(len(selector), len(codes))
    if not isinstance(selector, str):
        return 1
    return len(select_codes_from_pattern(codes, selector, dim_meta.get("label", "")))

def estimate_cell_count(metadata: dict, filters: dict) -> int:
    """Estimate number of cells (product of selected counts across dimensions)."""
    if not metadata or "dimension" not in metadata:
        return 0
    dims = metadata["dimension"]
    total = 1
    for dim_name, dim_meta in dims.items():
        # If eliminable and not provided → contributes 1 cell (eliminated)
        if dim_meta.get("elimination") and dim_name not in filters:
            total *= 1
            continue
        selector = filters.get(dim_name, "*")
        total *= max(1, count_selected_for_dim(dim_meta, selector))
    return total

def create_error_response(tool_name: str, table_id: str = None, error_msg: str = "", suggestion: str = "", **extras) -> dict:
    response = {
        "error": error_msg,
        "suggestion": suggestion,
        "agent_guidance": {"recommended_action": "retry_with_corrections", "tool_name": tool_name}
    }
    if table_id:
        response["table_id"] = table_id
    response.update(extras)
    return response

def enhance_variable_info(variables: list) -> list:
    return [
        {"display_name": v, "api_name": v, "is_mapped": False, "pattern_hint": f"Use '{v}' in API calls"}
        for v in variables
    ]

def add_agent_guidance(tool_name: str, result: dict, **context) -> dict:
    guidance_map = {
        "search_tables": {
            "next_suggested_tools": ["get_table_info"],
            "workflow_hint": f"Found {result.get('total_found', 0)} tables, consider analyzing the top-scored ones"
        },
        "get_table_info": {
            "next_suggested_tools": ["discover_dimension_values", "get_filtered_data"],
            "workflow_hint": "Complete table info ready - use api_name values for dimension queries"
        },
        "discover_dimension_values": {
            "next_suggested_tools": ["get_filtered_data"],
            "workflow_hint": "Use returned codes in filters for data retrieval"
        }
    }
    base = guidance_map.get(tool_name, {})
    result["agent_guidance"] = {"tool_name": tool_name, "status": "success" if "error" not in result else "error", **base, **context}
    return result

# =============================================================================
# SSB API TOOLS
# =============================================================================

@function_tool
async def search_tables(query: str, pastdays: int = 0, pagesize: int = 50) -> dict:
    """
    Search SSB tables (PxWebApi 2.0 beta) - simplified single query approach.
    Returns more results with basic relevance ordering.
    """
    args = {"query": query, "pastdays": pastdays, "pagesize": pagesize}
    try:
        base_url = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        lang = "no"
        # Increase default page size to get more results in one go
        page_size = min(max(1, pagesize), 100)

        # Single query - no variations, let SSB API handle the search
        params = {
            "query": query, 
            "lang": lang, 
            "pageSize": page_size,  # Get more results in one call
            "pagenumber": 1
        }
        
        if pastdays > 0:
            params["pastdays"] = pastdays

        # Single API call
        data = await robust_api_call(base_url, params)
        
        if not data:
            return {"error": "No data returned from API", "tables": [], "total_found": 0}
        
        # Check if it's an error response
        if isinstance(data, dict) and "error" in data:
            return {"error": data["error"], "suggestion": data.get("suggestion", ""), "tables": [], "total_found": 0}
        
        if not isinstance(data, dict) or "tables" not in data:
            return {"error": "Invalid response format from API", "tables": [], "total_found": 0}

        tables = data["tables"]
        
        # Transform to our format without re-sorting (SSB API already returns sorted results)
        result_tables = []
        
        for t in tables:
            # Basic fields
            table_info = {
                "id": t.get("id"),
                "title": t.get("label"),
                "description": t.get("description", ""),
                "updated": t.get("updated", ""),
                "time_period": f"{t.get('firstPeriod', '')} - {t.get('lastPeriod', '')}",
                "variables": len(t.get("variableNames", [])),
                "subject_area": t.get("subjectCode", ""),
                "paths": []
            }
            
            # Extract paths if available
            if "paths" in t:
                for p in t["paths"]:
                    if isinstance(p, dict):
                        table_info["paths"].append(p.get("id", ""))
                    elif isinstance(p, str):
                        table_info["paths"].append(p)
            
            # Check if it has recent data (for agent reference)
            last_period = str(t.get("lastPeriod", ""))
            current_year = datetime.now().year
            if str(current_year) in last_period or str(current_year - 1) in last_period:
                table_info["has_recent_data"] = True
            else:
                table_info["has_recent_data"] = False
            
            result_tables.append(table_info)
        
        result = add_agent_guidance("search_tables", {
            "query": query,
            "language": lang,
            "tables": result_tables,  # Return all results in SSB's order
            "total_found": len(result_tables),
            "search_tips": [
                "SSB API returnerer tabeller sortert etter relevans.",
                "Første treff er vanligvis det beste."
            ]
        })
        log_tool_call(LOG_FILE, "search_tables", args, result)
        return result
    except Exception as e:
        error_result = {"error": f"Search failed: {e}", "tables": [], "total_found": 0}
        log_tool_call(LOG_FILE, "search_tables", args, error_result)
        return error_result

@function_tool
async def get_table_info(table_id: str, include_structure: bool = True) -> dict:
    """
    Basic + detailed metadata (roles, valuesets/aggregations with codeList).
    """
    args = {"table_id": table_id, "include_structure": include_structure}
    try:
        base = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        info_url = f"{base}/{table_id}"
        meta_url = f"{base}/{table_id}/metadata"
        lang = "no"
        
        # Check cache first
        cache_key = (table_id, lang)
        if cache_key in _meta_cache and include_structure:
            meta = _meta_cache[cache_key]
        else:
            meta = None

        basic = await robust_api_call(info_url, {"lang": lang})
        if not basic:
            return create_error_response(
                "get_table_info", table_id, "Failed to retrieve table info",
                "Sjekk tabell-ID eller bruk search_tables."
            )
        
        # Check if it's an error response
        if isinstance(basic, dict) and "error" in basic:
            return create_error_response(
                "get_table_info", table_id, basic["error"],
                basic.get("suggestion", "Sjekk tabell-ID eller bruk search_tables.")
            )
        
        if not basic.get("label"):
            return create_error_response(
                "get_table_info", table_id, "Table not found or unavailable",
                "Sjekk tabell-ID eller bruk search_tables."
            )

        variables = basic.get("variableNames", [])
        result = {
            "table_id": table_id,
            "title": basic.get("label", ""),
            "description": basic.get("description", ""),
            "first_period": basic.get("firstPeriod", ""),
            "last_period": basic.get("lastPeriod", ""),
            "last_updated": basic.get("updated", ""),
            "source": "Statistics Norway (SSB)",
            "variables": enhance_variable_info(variables),
            "total_variables": len(variables),
            "time_span": f"{basic.get('firstPeriod','?')} - {basic.get('lastPeriod','?')}",
            "data_availability": {
                "has_recent_data": any(y in str(basic.get("lastPeriod", "")) for y in ("2024", "2025")),
                "time_coverage": basic.get("lastPeriod", "Unknown"),
                "frequency": "quarterly" if "K" in str(basic.get("lastPeriod","")) else
                             "monthly" if "M" in str(basic.get("lastPeriod","")) else "annual"
            }
        }

        if include_structure:
            if meta is None:
                meta = await robust_api_call(meta_url, {"lang": lang})
                if isinstance(meta, dict) and "error" not in meta:
                    _meta_cache[cache_key] = meta  # Cache the metadata
            if isinstance(meta, dict) and "error" not in meta:
                dims = meta.get("dimension", {}) or {}
                detailed = []
                agg_opts: Dict[str, Any] = {}
                for dim_name, dim_data in dims.items():
                    cat = dim_data.get("category", {})
                    # codes by position
                    codes = _codes_in_order(cat.get("index"))
                    labels = cat.get("label", {}) or {}
                    role = dim_data.get("role")

                    entry = {
                        "display_name": dim_name,
                        "api_name": dim_name,
                        "type": "dimension",
                        "role": role,
                        "total_values": len(codes),
                        "sample_values": codes[:5],
                        "sample_labels": [labels.get(c, c) for c in codes[:5]],
                    }

                    # ContentsCode → units/decimals
                    if dim_name == "ContentsCode":
                        ext = dim_data.get("extension", {})
                        contents = (ext.get("contents") or {})
                        units_info = {}
                        for c in codes[:10]:
                            if c in contents:
                                ci = contents[c]
                                units_info[c] = {"unit": ci.get("unit", ""), "decimals": ci.get("decimals", 0)}
                        if units_info:
                            entry["units_info"] = units_info

                    # valuesets / aggregations
                    ext = dim_data.get("extension", {})
                    code_lists = ext.get("codeLists", []) or []
                    if code_lists:
                        info = {"valuesets": [], "aggregations": []}
                        for item in code_lists:
                            t = (item.get("type") or "").lower()
                            payload = {"id": item.get("id", ""), "label": item.get("label", ""), "type": t}
                            if t == "valueset":
                                info["valuesets"].append(payload)
                            elif t == "aggregation":
                                info["aggregations"].append(payload)
                        if info["valuesets"] or info["aggregations"]:
                            agg_opts[dim_name] = info

                    detailed.append(entry)

                result["variables"] = detailed
                result["aggregation_options"] = agg_opts
                result["structure_included"] = True
            else:
                result["structure_included"] = False
        else:
            result["structure_included"] = False

        result["workflow_guidance"] = {
            "next_steps": [
                "Bruk discover_dimension_values for å finne riktige koder.",
                "Bruk get_filtered_data for uttrekk (valueCodes/codeList/outputValues)."
            ],
            "recommended_tools": ["discover_dimension_values", "get_filtered_data"],
            "workflow_hint": "Bruk eksakte API-navn (id) for dimensjoner."
        }
        result = add_agent_guidance("get_table_info", result)
        log_tool_call(LOG_FILE, "get_table_info", args, result)
        return result
    except Exception as e:
        error_result = create_error_response("get_table_info", table_id, f"Unexpected error: {e}", "Prøv igjen.")
        log_tool_call(LOG_FILE, "get_table_info", args, error_result)
        return error_result

@function_tool
async def discover_dimension_values(table_id: str, dimension_name: str, search_term: str = "", include_code_lists: bool = True, code_list: str = "") -> dict:
    """
    List values (code+label) for any dimension; optionally list codeList (valuesets/aggregations).
    Supports multi-term search (comma/space separated) and optional code_list narrowing.
    """
    args = {"table_id": table_id, "dimension_name": dimension_name, "search_term": search_term, "include_code_lists": include_code_lists, "code_list": code_list}
    try:
        meta_url = f"https://data.ssb.no/api/pxwebapi/v2-beta/tables/{table_id}/metadata"
        lang = "no"
        
        # Check cache first
        cache_key = (table_id, lang)
        if cache_key in _meta_cache:
            meta = _meta_cache[cache_key]
        else:
            meta = await robust_api_call(meta_url, {"lang": lang})
            if isinstance(meta, dict) and "error" not in meta:
                _meta_cache[cache_key] = meta
        
        if not meta:
            return create_error_response("discover_dimension_values", table_id, "Failed to retrieve metadata", "Sjekk tabell og dimensjonsnavn.")
        
        # Check if it's an error response
        if isinstance(meta, dict) and "error" in meta:
            return create_error_response("discover_dimension_values", table_id, meta["error"], meta.get("suggestion", "Sjekk tabell og dimensjonsnavn."))
        
        if "dimension" not in meta:
            return create_error_response("discover_dimension_values", table_id, "Invalid metadata format", "Sjekk tabell og dimensjonsnavn.")

        dims = meta["dimension"]
        if dimension_name not in dims:
            avail = list(dims.keys())
            return create_error_response("discover_dimension_values", table_id, f"Dimension '{dimension_name}' not found",
                                         f"Bruk en av: {avail}", available_dimensions=avail)

        dim = dims[dimension_name]
        cat = dim.get("category", {})
        codes_all = _codes_in_order(cat.get("index"))
        labels = cat.get("label", {}) or {}
        
        # Optional narrowing heuristic (generic)
        codes = codes_all
        if code_list:
            # Generic narrowing: keep codes matching certain patterns
            # Example: for municipalities, keep 4-digit numeric codes
            if dimension_name.lower() == "region" and "kommun" in code_list.lower():
                codes = [c for c in codes_all if c.isdigit() and len(c) == 4]
            elif dimension_name.lower() == "region" and "fylk" in code_list.lower():
                codes = [c for c in codes_all if c.isdigit() and len(c) == 2]
            else:
                codes = codes_all

        # Multi-term search support
        if search_term:
            terms = [t.strip().lower() for t in re.split(r"[,;\s]+", search_term) if t.strip()]
            def matches(c):
                lab = labels.get(c, c)
                s = f"{c} {lab}".lower()
                return any(t in s for t in terms)
            pairs = [(c, labels.get(c, c)) for c in codes if matches(c)]
        else:
            pairs = [(c, labels.get(c, c)) for c in codes]

        vals = [{"code": c, "label": lab} for c, lab in pairs]
        shown = vals[:20]
        truncated = max(0, len(vals) - len(shown))

        out = {
            "table_id": table_id,
            "dimension_name": dimension_name,
            "search_term": search_term,
            "total_values": len(codes),
            "matching_values": len(vals),
            "values": shown,
            "truncated_count": truncated,
            "usage_suggestion": f"Bruk 'code' i filters for {dimension_name}"
        }

        if include_code_lists:
            ext = dim.get("extension", {})
            code_lists = ext.get("codeLists", []) or []
            out["code_lists"] = {}
            out["recommendations"] = []
            for item in code_lists:
                cid = item.get("id", "")
                lbl = item.get("label", "")
                typ = (item.get("type") or "").lower()
                out["code_lists"][cid] = {"type": typ, "label": lbl, "usage_example": f"code_lists={{'{dimension_name}': '{cid}'}}"}
                if typ == "valueset":
                    out["recommendations"].append(f"Bruk '{cid}' for verdimengder (utvalg) i {dimension_name}.")
                elif typ == "aggregation":
                    out["recommendations"].append(f"Bruk '{cid}' for grupperinger/agg. i {dimension_name} (kombiner med outputValues).")

        result = add_agent_guidance("discover_dimension_values", out)
        log_tool_call(LOG_FILE, "discover_dimension_values", args, result)
        return result
    except Exception as e:
        error_result = create_error_response("discover_dimension_values", table_id, f"Failed: {e}", "Sjekk tabell og dimensjon.")
        log_tool_call(LOG_FILE, "discover_dimension_values", args, error_result)
        return error_result

def merge_table_data(table1_data: list, table2_data: list, join_keys: list) -> list:
    """Merge two record lists on join_keys (prefers *_code)."""
    def key_of(row):
        parts = []
        for k in join_keys:
            v = row.get(f"{k}_code")
            if v is None:
                v = row.get(k)
            parts.append("" if v is None else str(v))
        return "|".join(parts)

    lookup = {key_of(r): r for r in table2_data}
    merged = []
    for r1 in table1_data:
        k = key_of(r1)
        r2 = lookup.get(k)
        if r2:
            out = r1.copy()
            for f, v in r2.items():
                if f == "index":
                    continue
                if f == "value":
                    out["value_table2"] = v
                elif f not in out:
                    out[f] = v
            merged.append(out)
    return merged

@function_tool
async def merge_tables(table1_id: str, table1_filters_json: str, table2_id: str, table2_filters_json: str, join_dimensions_json: str) -> dict:
    """
    Retrieve two tables and merge on specified dimensions (e.g., ["Tid","Region"]).
    """
    try:
        join_dims = json.loads(join_dimensions_json) if join_dimensions_json else []
        if not join_dims:
            return {"error": "Join dimensions required", "suggestion": "Angi felles dimensjoner, f.eks. ['Tid','Region']."}

        d1 = await get_filtered_data(table1_id, table1_filters_json)
        if "error" in d1:
            return {"error": f"Table1 failed: {d1['error']}", "suggestion": d1.get("suggestion", "")}
        d2 = await get_filtered_data(table2_id, table2_filters_json)
        if "error" in d2:
            return {"error": f"Table2 failed: {d2['error']}", "suggestion": d2.get("suggestion", "")}

        r1, r2 = d1.get("formatted_data", []), d2.get("formatted_data", [])
        if not r1 or not r2:
            return {"error": "No data to merge", "suggestion": "Sjekk filter for begge tabeller."}

        merged = merge_table_data(r1, r2, join_dims)
        out = {
            "table1_id": table1_id,
            "table1_title": d1.get("title", ""),
            "table2_id": table2_id,
            "table2_title": d2.get("title", ""),
            "join_dimensions": join_dims,
            "table1_rows": len(r1),
            "table2_rows": len(r2),
            "merged_rows": len(merged),
            "merged_data": merged[:200],
            "truncated": len(merged) > 200,
        }

        if merged:
            v1 = [x["value"] for x in merged if isinstance(x.get("value"), (int, float))]
            v2 = [x["value_table2"] for x in merged if isinstance(x.get("value_table2"), (int, float))]
            if v1:
                out["table1_stats"] = {"min": min(v1), "max": max(v1), "avg": sum(v1)/len(v1)}
            if v2:
                out["table2_stats"] = {"min": min(v2), "max": max(v2), "avg": sum(v2)/len(v2)}
        return out
    except Exception as e:
        return {"error": f"Merge failed: {e}", "suggestion": "Sjekk parametre og at tabellene deler dimensjoner."}

@function_tool
async def get_filtered_data(
    table_id: str,
    filters_json: str,
    time_selection: str = "",
    code_lists_json: str = "",
    output_values_json: str = "",
    max_points: int = 500
) -> dict:
    """
    Retrieve filtered data using PxWebApi 2.0 beta GET (preferred) or POST (if URL too long).
    - filters_json: {"Region": "0301,3806" | "*" | "top(3)" | ["0301","3806"], "Tid":"from(2015)", ...}
    - code_lists_json: {"Region": "agg_Fylker2024" | "vs_Fylker" | "agg_KommSummer", ...}
    - output_values_json: {"Region": "aggregated"|"single"}  (use with aggregations/valuesets)
    - max_points: cap number of decoded records (display/sanity)
    """
    args = {
        "table_id": table_id, 
        "filters_json": filters_json,
        "time_selection": time_selection,
        "code_lists_json": code_lists_json,
        "output_values_json": output_values_json,
        "max_points": max_points
    }
    try:
        filters = json.loads(filters_json) if filters_json else {}
        code_lists = json.loads(code_lists_json) if code_lists_json else {}
        output_values = json.loads(output_values_json) if output_values_json else {}
        if not isinstance(filters, dict) or not filters:
            return create_error_response(
                "get_filtered_data", table_id,
                "filters parameter is required and must be a dict",
                "Bruk discover_dimension_values for å finne riktige koder, f.eks. {'Region':'0301','Tid':'2024'}."
            )

        base = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        data_url = f"{base}/{table_id}/data"
        meta_url = f"{base}/{table_id}/metadata"
        lang = "no"
        # Safety: estimate cells
        metadata = await robust_api_call(meta_url, {"lang": lang})
        if metadata:
            cells = estimate_cell_count(metadata, filters)
            if cells > 800_000:
                return create_error_response(
                    "get_filtered_data", table_id,
                    f"Query estimated to return {cells:,} cells (> 800,000 limit).",
                    "Reduser omfanget (tid/dimensjoner) eller bruk verdimengder/aggregat."
                )

        # Build GET params (preferred)
        params = {"lang": lang, "outputformat": "json-stat2"}
        for dim, cl in (code_lists or {}).items():
            params[f"codeList[{dim}]"] = cl
        for dim, ov in (output_values or {}).items():
            params[f"outputValues[{dim}]"] = ov

        # valueCodes
        for dim, sel in filters.items():
            if isinstance(sel, list):
                params[f"valueCodes[{dim}]"] = ",".join(str(v) for v in sel)
            else:
                params[f"valueCodes[{dim}]"] = str(sel)

        # Legacy convenience
        if time_selection and "Tid" not in filters:
            params["valueCodes[Tid]"] = time_selection

        url_with_params = data_url + "?" + urlencode(params, safe="[],(),*")
        use_post = _needs_post(filters) or len(url_with_params) > 2000

        if not use_post:
            data = await robust_api_call(data_url, params)
        else:
            # POST: keep codeList/outputValues in query string, put selections as explicit lists in body
            body = {"lang": lang, "outputformat": "json-stat2", "query": []}
            dims_meta = (metadata or {}).get("dimension", {}) if isinstance(metadata, dict) else {}
            for dim, sel in filters.items():
                # Expand any patterns to explicit code arrays (necessary for POST)
                codes_all = _codes_in_order(dims_meta.get(dim, {}).get("category", {}).get("index", []))
                if isinstance(sel, list):
                    values = [str(v) for v in sel]
                else:
                    values = select_codes_from_pattern(codes_all, str(sel), dim)
                body["query"].append({"code": dim, "selection": {"filter": "item", "values": values}})
            # carry codeList/outputValues in query params to keep semantics identical to GET
            post_params = {"lang": lang, "outputformat": "json-stat2"}
            for dim, cl in (code_lists or {}).items():
                post_params[f"codeList[{dim}]"] = cl
            for dim, ov in (output_values or {}).items():
                post_params[f"outputValues[{dim}]"] = ov
            data = await robust_api_call(data_url, params=post_params, json_body=body)

        if not data:
            return create_error_response("get_filtered_data", table_id, "Failed to fetch data", "Sjekk filter og prøv igjen.")

        if isinstance(data, dict) and "error" in data:
            out = create_error_response(
                "get_filtered_data", table_id,
                f"SSB API error: {data['error']}",
                "Sjekk dimensjonsnavn og koder (discover_dimension_values).",
                filters_attempted=filters
            )
            if "details" in data:
                out["api_details"] = data["details"]
            return out

        # Success → decode
        recs = jsonstat_to_records(data, max_points=max_points)
        # Attach unit/decimals metadata per ContentsCode (no rescaling)
        dims = data.get("dimension", {}) or {}
        if "ContentsCode" in dims:
            ext = (dims["ContentsCode"].get("extension") or {})
            contents = ext.get("contents") or {}
            for r in recs:
                cc = r.get("ContentsCode_code")
                if cc and cc in contents:
                    ci = contents[cc]
                    r["unit"] = ci.get("unit", "")
                    r["decimals"] = ci.get("decimals", 0)

        # Summary (based on raw value array)
        vals = [v for v in data.get("value", []) if isinstance(v, (int, float))]
        summary = None
        if vals:
            summary = {"count": len(vals), "min": min(vals), "max": max(vals), "average": (sum(vals) / len(vals)) if vals else None}

        # Consistent counting
        observations_total = len(data.get("value", []))
        observations_returned = len(recs)
        
        result = {
            "table_id": table_id,
            "title": data.get("label", f"Table {table_id}"),
            "source": data.get("source", "Statistics Norway"),
            "updated": data.get("updated", ""),
            "filters_applied": filters,
            "dimensions": list(dims.keys()),
            "observations_total": observations_total,
            "observations_returned": observations_returned,
            "formatted_data": recs,
            **({"summary_stats": summary} if summary else {})
        }
        
        # Always log before returning
        log_tool_call(LOG_FILE, "get_filtered_data", args, result)
        return result
    except Exception as e:
        error_result = create_error_response("get_filtered_data", table_id, f"Unexpected error: {e}", "Sjekk parametre og prøv igjen.", filters_applied=filters)
        log_tool_call(LOG_FILE, "get_filtered_data", args, error_result)
        return error_result

# =============================================================================
# Standalone Agent (console UX)
# =============================================================================

class SSBStandaloneAgent:
    """Standalone SSB Agent with direct function calls - no MCP needed."""

    def __init__(self):
        # .env loading
        simple_ssb_env = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
        if os.path.exists(simple_ssb_env):
            load_dotenv(simple_ssb_env, override=True)
        else:
            env_path = find_dotenv(filename=".env", usecwd=True)
            if env_path:
                load_dotenv(env_path, override=True)

        set_tracing_disabled(True)

        # Azure OpenAI client (Responses API)
        azure_client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        set_default_openai_client(azure_client)
        set_default_openai_api("responses")

        self.model = os.getenv("AZURE_OPENAI_MODEL", model)

    # --- Console renderers ----------------------------------------------------

    def _display_tool_output(self, tool_name: str, output: Any) -> None:
        console.print(f"[bold cyan]📤 {tool_name} Result:[/bold cyan]")
        try:
            if isinstance(output, dict):
                if "error" in output:
                    console.print(f"[red]❌ Error: {output['error']}[/red]")
                    if "suggestion" in output:
                        console.print(f"[blue]💡 Suggestion: {output['suggestion']}[/blue]")
                    return
                if tool_name == "search_tables":
                    self._display_search_results(output)
                elif tool_name == "get_table_info":
                    self._display_table_analysis(output)
                elif tool_name == "discover_dimension_values":
                    self._display_dimension_values(output)
                elif tool_name == "get_filtered_data":
                    self._display_filtered_data(output)
                else:
                    self._display_generic_output(output)
            else:
                console.print(f"[white]{output}[/white]")
        except Exception as e:
            console.print(f"[red]Error displaying output: {e}[/red]")
            console.print(f"[dim]Raw output: {output}[/dim]")

    def _display_search_results(self, data: dict) -> None:
        query = data.get("query", "Unknown")
        tables = data.get("tables", [])
        console.print(f"[green]🔍 Fant {len(tables)} tabeller for: '{query}'[/green]")
        if tables:
            t = Table(title="Search Results")
            t.add_column("ID", style="cyan")
            t.add_column("Title", style="white", max_width=60)
            t.add_column("Updated", style="dim")
            t.add_column("Score", style="green")
            for row in tables[:5]:
                t.add_row(row.get("id","N/A"), row.get("title","N/A"), str(row.get("updated","N/A")), f"{row.get('score','N/A')}")
            console.print(t)

    def _display_table_analysis(self, data: dict) -> None:
        tid = data.get("table_id", "Unknown")
        title = data.get("title", "Unknown")
        console.print(f"[green]📊 Table Analysis: {tid}[/green]")
        console.print(f"[white]{title}[/white]")
        fp, lp = data.get("first_period",""), data.get("last_period","")
        if fp or lp:
            console.print(f"[dim]📅 Tidsdekning: {fp} – {lp}[/dim]")
        vars_ = data.get("variables", [])
        if vars_:
            console.print(f"[cyan]📋 Dimensjoner ({len(vars_)}):[/cyan]")
            for v in vars_[:5]:
                console.print(f"  • {v.get('api_name','?')} ({v.get('total_values','?')} verdier)")

    def _display_dimension_values(self, data: dict) -> None:
        dim = data.get("dimension_name", "Unknown")
        tot = data.get("total_values", 0)
        shown = len(data.get("values", []))
        console.print(f"[green]🎯 Dimensjon: {dim}[/green]")
        console.print(f"[dim]Viser {shown} av {tot} verdier[/dim]")
        vals = data.get("values", [])
        if vals:
            t = Table(show_header=True, box=None)
            t.add_column("Code", style="cyan")
            t.add_column("Label", style="white", max_width=50)
            for v in vals[:10]:
                t.add_row(v.get("code",""), v.get("label",""))
            console.print(t)

    def _display_filtered_data(self, data: dict) -> None:
        tid = data.get("table_id", "Unknown")
        title = data.get("title", "Unknown")
        tot = data.get("observations_total", data.get("total_data_points", 0))
        ret = data.get("observations_returned", data.get("returned_data_points", 0))
        console.print(f"[green]📈 Data fra {tid}[/green]")
        console.print(f"[white]{title}[/white]")
        console.print(f"[dim]Punkter: {ret} av {tot}[/dim]")
        ss = data.get("summary_stats", {})
        if ss:
            console.print(f"[cyan]📊 Sammendrag:[/cyan]")
            for k, v in ss.items():
                console.print(f"  • {k}: {v:,.2f}" if isinstance(v, float) else f"  • {k}: {v:,}")
        sample = data.get("formatted_data", [])
        if sample:
            console.print(f"[cyan]📋 Eksempel:[/cyan]")
            for row in sample[:5]:
                val = row.get("value")
                dims = {k: v for k, v in row.items() if k not in {"value", "index"}}
                dim_str = ", ".join([f"{k}={v}" for k, v in dims.items() if not k.endswith("_code")])
                console.print(f"  • {val} ({dim_str})")

    def _display_generic_output(self, data: dict) -> None:
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                console.print(f"[cyan]{k}:[/cyan] [dim]{type(v).__name__} with {len(v)} items[/dim]")
            else:
                console.print(f"[cyan]{k}:[/cyan] [white]{v}[/white]")

    def _parse_tool_output(self, output: Any) -> Any:
        try:
            if isinstance(output, dict) and "content" in output:
                content = output["content"]
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and "text" in first:
                        try:
                            return json.loads(first["text"])
                        except json.JSONDecodeError:
                            return first["text"]
                    return content
                return content
            if isinstance(output, dict) and output.get("type") == "text":
                t = output.get("text", "")
                try:
                    return json.loads(t)
                except json.JSONDecodeError:
                    return t
            if isinstance(output, str):
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return output
            return output
        except Exception:
            return output

    def _display_tool_result_summary(self, tool_name: str, output: Any) -> None:
        try:
            parsed = self._parse_tool_output(output)
            if isinstance(parsed, dict):
                if "error" in parsed:
                    console.print(f"   [red]❌ {parsed.get('error')}[/red]")
                    return
                if tool_name == "search_tables":
                    tabs = parsed.get("tables", [])
                    total = parsed.get("total_found", len(tabs))
                    if tabs:
                        console.print(f"   [green]✓ Found {total} tables[/green]")
                        for i, t in enumerate(tabs[:5]):
                            tid, title = t.get("id","N/A"), (t.get("title","N/A") or "")[:60]
                            if title.startswith(f"{tid}: "): title = title[len(f"{tid}: "):]
                            console.print(f"     {i+1}. [cyan]{tid}[/cyan] - {title}...")
                        if len(tabs) > 5:
                            console.print(f"     [dim]... and {len(tabs)-5} more[/dim]")
                    else:
                        console.print("   [yellow]⚠ No tables found[/yellow]")
                elif tool_name == "get_table_info":
                    tid = parsed.get("table_id", "N/A")
                    vars_ = parsed.get("variables", [])
                    span = parsed.get("time_span", "N/A")
                    console.print(f"   [green]✓ Table {tid}: {len(vars_)} dims, {span}[/green]")
                elif tool_name == "discover_dimension_values":
                    dim = parsed.get("dimension_name", "N/A")
                    total = parsed.get("total_values", 0)
                    match = parsed.get("matching_values", 0)
                    search = parsed.get("search_term", "")
                    values = parsed.get("values", [])
                    
                    if search:
                        console.print(f"   [green]✓ Dimension '{dim}': Found {match} matches for '{search}'[/green]")
                    else:
                        console.print(f"   [green]✓ Dimension '{dim}': {match}/{total} values[/green]")
                    
                    # Show first few found values
                    if values:
                        for v in values[:3]:
                            code = v.get("code", "")
                            label = v.get("label", "")
                            console.print(f"     • {code}: {label}")
                        if len(values) > 3:
                            console.print(f"     [dim]... and {len(values)-3} more[/dim]")
                elif tool_name == "get_filtered_data":
                    tid = parsed.get("table_id", "N/A")
                    tot = parsed.get("observations_total", parsed.get("total_data_points", 0))
                    ret = parsed.get("observations_returned", parsed.get("returned_data_points", 0))
                    console.print(f"   [green]✓ Retrieved {ret}/{tot} observations from {tid}[/green]")
                else:
                    keys = [k for k in parsed.keys() if k not in {"error","agent_guidance"}][:3]
                    console.print(f"   [green]✓ Returned fields: {', '.join(keys)}[/green]")
            else:
                console.print(f"   [green]✓ {tool_name} completed[/green]")
        except Exception:
            console.print(f"   [yellow]⚠ {tool_name} completed (summary unavailable)[/yellow]")

    async def process_query(self, query: str) -> str:
        """Run the agent with streaming output."""
        t0 = time.monotonic()
        try:
            agent = Agent(
                name="SSB Statistikk-ekspert",
                instructions=(
                    "Du er en ekspert på norsk statistikk og bruker SSB PxWebApi 2.0 beta.\n\n"
                    "OPTIMALISERT ARBEIDSFLYT (minimer verktøykall):\n\n"
                    "1. MINIMAL PATH - Typisk trenger du bare 3 kall:\n"
                    "   a) search_tables - ETT kall med konsept-orientert query\n"
                    "   b) get_table_info(include_structure=True) - får all metadata inkludert koder\n"
                    "   c) get_filtered_data - ETT kall med alle nødvendige filtre\n\n"
                    "2. HOPP OVER STEG når mulig:\n"
                    "   - SKIP search_tables hvis du kjenner tabell-ID\n"
                    "   - SKIP discover_dimension_values hvis du kan løse koder fra metadata\n"
                    "   - Bruk metadata fra get_table_info for å finne koder (sjekk category.label)\n\n"
                    "3. NÅR bruke discover_dimension_values:\n"
                    "   - KUN når metadata-oppslag feiler\n"
                    "   - Ved usikre verdier (f.eks. ukjente kategorikoder)\n"
                    "   - For fuzzy matching på mange labels\n"
                    "   - Støtter multi-term søk: search_term='Oslo Porsgrunn Bergen'\n\n"
                    "4. METADATA RESOLUTION (unngå ekstra kall):\n"
                    "   - get_table_info gir deg category.label mappinger\n"
                    "   - Bruk disse for å løse labels→koder direkte\n"
                    "   - Eksempel: 'Oslo' finnes i metadata → kode '0301'\n\n"
                    "5. KOMBINER verdier i ETT kall:\n"
                    "   - Flere regioner: {'Region': '0301,5001,4601'} eller ['0301','5001','4601']\n"
                    "   - Ikke separate kall for hver region!\n\n"
                    "6. POST brukes automatisk når:\n"
                    "   - Filtre er lister eller inneholder mønstre (top, from, range, *, ?)\n"
                    "   - URL blir for lang (>2000 tegn)\n\n"
                    "TEKNISK:\n"
                    "- filters_json er PÅKREVD (f.eks. '{\"Tid\":\"2024\",\"Region\":\"0301\"}')\n"
                    "- Tidsperioder: from(år), top(n), [range(start,slutt)]\n"
                    "- For aggregering: code_lists_json + output_values_json\n"
                    "- Metadata caches automatisk\n\n"
                    "SVAR PÅ NORSK med tabellkilder.\n"
                ),
                model=self.model,
                tools=[search_tables, get_table_info, discover_dimension_values, get_filtered_data, merge_tables],
                model_settings=agent_model_settings.ModelSettings(
                    reasoning={"effort": os.getenv("AZURE_REASONING_EFFORT", "medium"),
                               "summary": os.getenv("AZURE_REASONING_SUMMARY", "auto")}
                ),
            )

            console.print(f"[bold blue]🧠 {self.model} Analysis[/bold blue]\n")
            result = run.Runner.run_streamed(agent, query, max_turns=20)

            tool_calls = []
            final_chunks = []
            first_token_time = None

            with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), transient=True) as progress:
                task = progress.add_task("Processing...", total=None)

                async for event in result.stream_events():
                    if event.type == "run_item_stream_event":
                        if event.item.type == "tool_call_item":
                            tool_name = getattr(getattr(event.item, "raw_item", None), "name", "unknown_tool")
                            args = {}
                            try:
                                raw_args = getattr(getattr(event.item, "raw_item", None), "arguments", {})
                                if isinstance(raw_args, str):
                                    args = json.loads(raw_args)
                                else:
                                    args = raw_args or {}
                            except Exception:
                                pass
                            tool_calls.append(tool_name)

                            progress.stop()
                            console.print(f"\n[yellow]🔧 Calling: {tool_name}[/yellow]")
                            if args:
                                if tool_name == "search_tables" and "query" in args:
                                    console.print(f"   [dim]→ Query: {args['query']}[/dim]")
                                elif tool_name == "get_table_info" and "table_id" in args:
                                    console.print(f"   [dim]→ Table: {args['table_id']}[/dim]")
                                elif tool_name == "discover_dimension_values":
                                    if "dimension_name" in args:
                                        console.print(f"   [dim]→ Dimension: {args['dimension_name']}[/dim]")
                                    if "search_term" in args and args.get("search_term"):
                                        console.print(f"   [dim]→ Searching for: {args['search_term']}[/dim]")
                                elif tool_name == "get_filtered_data":
                                    if "table_id" in args:
                                        console.print(f"   [dim]→ Table: {args['table_id']}[/dim]")
                                    if "filters_json" in args:
                                        try:
                                            f = json.loads(args["filters_json"])
                                            console.print(f"   [dim]→ Filters: {json.dumps(f, ensure_ascii=False)}[/dim]")
                                        except Exception:
                                            pass
                            progress.start()
                            progress.update(task, description=f"Executing {tool_name}...")
                        elif event.item.type == "tool_call_output_item":
                            progress.stop()
                            self._display_tool_result_summary(tool_calls[-1] if tool_calls else "tool", event.item.output)
                            progress.start()
                            progress.update(task, description="Processing results...")
                        elif event.item.type == "message_output_item":
                            progress.stop()
                    elif event.type == "raw_response_event":
                        if hasattr(event.data, "type") and event.data.type == "response.output_text.delta":
                            if first_token_time is None:
                                first_token_time = time.monotonic()
                                progress.stop()
                                console.print("\n[bold green]📋 Answer:[/bold green]")
                            if getattr(event.data, "delta", None):
                                final_chunks.append(event.data.delta)
                                console.print(f"[white]{event.data.delta}[/white]", end="")
                        elif hasattr(event.data, "type") and event.data.type == "response.output_text.done":
                            console.print("")

            final_output = result.final_output or "".join(final_chunks)
            return final_output
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")
            return f"Error: {e}"
        finally:
            console.print("\n" + "="*60)
            console.print("🎯 Standalone Agent Summary")
            console.print(f"⏱ Total time: {int((time.monotonic() - t0)*1000)} ms")
            console.print("="*60)

# =============================================================================
# CLI
# =============================================================================

async def main():
    if len(sys.argv) != 2:
        console.print("[red]Bruk:[/red] python ssb_standalone_agent.py \"Din spørring\"")
        console.print("\n[bold]Eksempler:[/bold]")
        console.print("\n[cyan]Grunnleggende spørringer:[/cyan]")
        console.print("  python ssb_standalone_agent.py \"befolkning i Norge\"")
        console.print("  python ssb_standalone_agent.py \"arbeidsledighet etter region\"")
        console.print("  python ssb_standalone_agent.py \"utdanningsnivå fylkesvis\"")
        console.print("\n[cyan]Sammenligning og analyse:[/cyan]")
        console.print("  python ssb_standalone_agent.py \"sammenlign befolkning Oslo og Bergen\"")
        console.print("  python ssb_standalone_agent.py \"arbeidsledighet utvikling siden 2020\"")
        console.print("  python ssb_standalone_agent.py \"nyeste utdanningsstatistikk per region\"")
        console.print("\n[cyan]FoU og spesialiserte domener:[/cyan]")
        console.print("  python ssb_standalone_agent.py \"hvor mye av fou utgifter finansiert av eu\"")
        console.print("  python ssb_standalone_agent.py \"hvilken næring har flest sysselsatte\"")
        console.print("\n[dim]💡 Tips: Norske nøkkelord gir best resultater fra SSB[/dim]")
        sys.exit(1)

    query = sys.argv[1]
    console.print(Panel(Text(query, style="bold cyan"), title="[bold blue]🤖 SSB Standalone Agent[/bold blue]", border_style="blue"))
    console.print()

    agent = SSBStandaloneAgent()
    _ = await agent.process_query(query)

if __name__ == "__main__":
    asyncio.run(main())