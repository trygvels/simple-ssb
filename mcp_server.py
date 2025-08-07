#!/usr/bin/env python3
"""
Enhanced SSB MCP Server - Advanced filtering, aggregation, and data analysis
"""

import asyncio
import httpx
from typing import Dict, List, Any, Optional
from fastmcp import FastMCP
from pydantic import Field
from datetime import datetime, timedelta
import json
import hashlib
import re

# Create FastMCP server
mcp = FastMCP("Enhanced SSB Discovery")

def generate_advanced_ssb_queries(original_query: str) -> List[str]:
    """
    Generate advanced SSB API queries using official search syntax.
    
    Based on SSB API documentation:
    - Use title: prefix to search in titles only
    - Use AND/OR/NOT operators for complex searches  
    - Use (K), (F), (T) suffixes for administrative levels
    - Exclude discontinued series with NOT "closed series"
    - Use Norwegian terms for better results
    """
    
    query = original_query.lower().strip()
    variations = []
    
    # Start with original query (fallback)
    variations.append(original_query)
    
    # Extract meaningful words (3+ characters, skip common words)
    stop_words = {'hvor', 'mange', 'hva', 'når', 'the', 'and', 'or', 'in', 'på', 'med', 'for', 'til', 'av', 'og', 'i'}
    words = [w for w in query.split() if len(w) > 2 and w not in stop_words]
    
    if words:
        # Use the longest word as primary term (often the most specific)
        primary_term = max(words, key=len)
        
        # Generate systematic variations using SSB search syntax
        variations.extend([
            f'title:{primary_term} NOT "closed series"',
            f'title:{primary_term} AND title:(K) NOT "closed series"',  # Municipal
            f'title:{primary_term} AND title:(F) NOT "closed series"',  # County
            primary_term  # Broad fallback
        ])
        
        # If multiple meaningful words, try combinations
        if len(words) > 1:
            # Try with second most relevant word
            secondary_terms = [w for w in words if w != primary_term]
            if secondary_terms:
                secondary_term = max(secondary_terms, key=len)
                variations.append(f'title:{primary_term} AND title:{secondary_term} NOT "closed series"')
    
    # Remove duplicates while preserving order, limit for efficiency
    seen = set()
    final_variations = []
    for var in variations:
        if var and var not in seen and len(final_variations) < 4:
            seen.add(var)
            final_variations.append(var)
    
    return final_variations

# Rate limiter and cache classes
class RateLimiter:
    def __init__(self, max_calls=25, time_window=600):  # Conservative limit
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        now = datetime.now()
        # Remove old calls outside time window
        self.calls = [call for call in self.calls 
                     if call > now - timedelta(seconds=self.time_window)]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]).total_seconds()
            if sleep_time > 0:
                # Cap wait time to prevent MCP timeouts
                await asyncio.sleep(min(sleep_time, 2.0))
        
        self.calls.append(now)

# Global instances
rate_limiter = RateLimiter()

async def robust_api_call(url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
    """Make API call with retry logic and rate limiting."""
    await rate_limiter.acquire()
    
    print(f"DEBUG: Calling API {url} with params {params}")
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                print(f"DEBUG: API response status: {response.status_code}")
                response.raise_for_status()
                result = response.json()
                print(f"DEBUG: API returned {type(result)} with keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                return result
                
        except httpx.HTTPStatusError as e:
            print(f"DEBUG: HTTP error {e.response.status_code}")
            error_details = {
                "status_code": e.response.status_code,
                "url": str(e.request.url),
                "params": params,
                "response_text": e.response.text[:500] if hasattr(e.response, 'text') else "N/A"
            }
            print(f"DEBUG: Error details: {error_details}")
            
            if e.response.status_code == 429:  # Rate limited
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                # Return detailed error instead of raising
                return {
                    "error": f"HTTP {e.response.status_code} error from SSB API",
                    "details": error_details,
                    "suggestion": "Check API parameters and table availability"
                }
                
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            print(f"DEBUG: Request/Timeout error: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    
    print("DEBUG: All retries exhausted, returning None")
    return None

async def robust_api_call_post(url: str, json_body: dict, max_retries: int = 3) -> Optional[dict]:
    """Make POST API call with retry logic and rate limiting."""
    await rate_limiter.acquire()
    
    print(f"DEBUG: Calling POST API {url} with body {json_body}")
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=json_body)
                print(f"DEBUG: POST API response status: {response.status_code}")
                response.raise_for_status()
                result = response.json()
                print(f"DEBUG: POST API returned {type(result)} with keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                return result
                
        except httpx.HTTPStatusError as e:
            print(f"DEBUG: POST HTTP error {e.response.status_code}")
            error_details = {
                "status_code": e.response.status_code,
                "url": str(e.request.url),
                "json_body": json_body,
                "response_text": e.response.text[:500] if hasattr(e.response, 'text') else "N/A"
            }
            print(f"DEBUG: POST Error details: {error_details}")
            
            if e.response.status_code == 429:  # Rate limited
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                # Return detailed error instead of raising
                return {
                    "error": f"HTTP {e.response.status_code} error from SSB API",
                    "details": error_details,
                    "suggestion": "Check API parameters and table availability"
                }
                
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            print(f"DEBUG: POST Request/Timeout error: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    
    print("DEBUG: POST All retries exhausted, returning None")
    return None

@mcp.tool()
async def search_tables_advanced(
    query: str = Field(description="Search query for SSB statistical tables"),
    language: str = Field(default="no", pattern="^(no|en)$", description="Language for results"),
    max_results: int = Field(default=20, le=50, description="Maximum number of results"),
    recent_only: bool = Field(default=False, description="Only recently updated tables")
) -> dict:
    """
    Advanced search for SSB statistical tables using official API search syntax.
    
    Leverages SSB's powerful search features:
    - title: prefix to search in titles only
    - AND/OR/NOT operators for precise filtering
    - (K), (F), (T) suffixes for administrative levels
    - Excludes discontinued series with NOT "closed series"
    - Multiple query variations for comprehensive coverage
    
    This ensures highly relevant, active tables are found efficiently.
    """
    
    try:
        print(f"DEBUG: Starting multi-query search for '{query}'")
        
        # Generate advanced SSB API queries using official syntax
        query_variations = generate_advanced_ssb_queries(query)
        print(f"DEBUG: Generated variations: {query_variations}")
        
        base_url = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        all_tables = {}  # Use dict to deduplicate by table ID
        
        # Try each query variation with timeout protection
        max_variations = min(3, len(query_variations))  # Limit to 3 variations for speed
        for i, search_query in enumerate(query_variations[:max_variations], 1):
            print(f"DEBUG: Trying variation {i}/{max_variations}: '{search_query}'")
            
            params = {
                "query": search_query,
                "lang": language,
                "pageSize": min(max_results, 50)
            }
            
            if recent_only:
                params["pastDays"] = 30
            
            # Direct API call without caching to avoid cache pollution
            data = await robust_api_call(base_url, params)
            
            if data and 'tables' in data:
                variation_tables = data.get('tables', [])
                print(f"DEBUG: Variation '{search_query}' found {len(variation_tables)} tables")
                
                # Add tables to collection, avoiding duplicates
                for table in variation_tables:
                    table_id = table.get('id')
                    if table_id and table_id not in all_tables:
                        all_tables[table_id] = table
                        
                # If we have enough results, stop early
                if len(all_tables) >= max_results:
                    print(f"DEBUG: Found enough tables ({len(all_tables)}), stopping early")
                    break
            
            # Reduced delay for speed
            if i < max_variations:
                await asyncio.sleep(0.05)
        
        tables = list(all_tables.values())
        print(f"DEBUG: Multi-query search found {len(tables)} unique tables total")
        
        # Simple but effective scoring for population queries
        query_words = query.lower().split()
        scored_tables = []
        
        for table in tables:
            label = table.get('label', '').lower()
            description = table.get('description', '').lower()
            table_id = table.get('id', '')
            
            # Start with base score of 1 to ensure all tables are considered
            score = 1
            
            # Domain-agnostic scoring based on term matches
            for word in query_words:
                if len(word) > 2:  # Skip short words
                    if word in label:
                        score += 10
                    if word in description:
                        score += 5
            
            # Boost for recent updates (fresher data preferred)
            updated = table.get('updated', '')
            if updated and ('2024' in updated or '2025' in updated):
                score += 5
                    
            scored_tables.append({
                'id': table.get('id'),
                'title': table.get('label'),
                'description': table.get('description', ''),
                'updated': table.get('updated', ''),
                'score': score,
                'time_period': f"{table.get('firstPeriod', '')} - {table.get('lastPeriod', '')}",
                'variables': len(table.get('variableNames', table.get('variables', []))),  # Support both formats
                'subject_area': table.get('subjectCode', '')
            })
        
        # Sort by score
        scored_tables.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"DEBUG: Returning {len(scored_tables)} scored tables")
        
        return {
            "query": query,
            "language": language,
            "tables": scored_tables[:max_results],
            "total_found": len(scored_tables),
            "search_tips": [
                "Try Norwegian keywords for better results (e.g., 'befolkning' instead of 'population')",
                "Use specific terms (e.g., 'arbeidsledighet' rather than 'jobb')",
                "Check different time periods with recent_only filter"
            ]
        }
        
    except Exception as e:
        return {"error": f"Advanced search failed: {str(e)}", "tables": [], "total_found": 0}

@mcp.tool()
async def get_table_info(
    table_id: str = Field(description="SSB table identifier"),
    language: str = Field(default="no", pattern="^(no|en)$", description="Language preference")
) -> dict:
    """
    Get basic table information including title, time coverage, and variables.
    
    Uses the /tables/{id} endpoint for quick overview before detailed analysis.
    Shows firstPeriod, lastPeriod, variableNames, and subject classification.
    """
    
    try:
        print(f"DEBUG: Getting basic info for table {table_id}")
        
        base_url = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        info_url = f"{base_url}/{table_id}"
        
        params = {"lang": language}
        
        data = await robust_api_call(info_url, params)
        if not data:
            return {"error": "Failed to retrieve table information", "table_id": table_id}
        
        info = {
            "table_id": table_id,
            "title": data.get("label", "No title"),
            "description": data.get("description", ""),
            "first_period": data.get("firstPeriod", "Unknown"),
            "last_period": data.get("lastPeriod", "Unknown"),
            "last_updated": data.get("updated", "Unknown"),
            "variables": data.get("variableNames", []),
            "subject_paths": data.get("path", []),
            "total_variables": len(data.get("variableNames", [])),
            "time_span": f"{data.get('firstPeriod', 'Unknown')} - {data.get('lastPeriod', 'Unknown')}"
        }
        
        return info
        
    except Exception as e:
        print(f"DEBUG: Error getting table info: {str(e)}")
        return {"error": f"Failed to get table info: {str(e)}", "table_id": table_id}

@mcp.tool()
async def analyze_table_structure(
    table_id: str = Field(description="SSB table identifier"),
    language: str = Field(default="no", pattern="^(no|en)$", description="Language preference")
) -> dict:
    """Analyze table structure and provide detailed metadata with aggregation options and query suggestions."""
    
    try:
        base_url = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        metadata_url = f"{base_url}/{table_id}/metadata"
        params = {"lang": language}
        
        metadata = await robust_api_call(metadata_url, params)
        if not metadata:
            return {"error": "Failed to fetch metadata", "table_id": table_id}
        
        analysis = {
            "table_id": table_id,
            "title": metadata.get("label", ""),
            "description": metadata.get("description", ""),
            "last_updated": metadata.get("updated", ""),
            "source": metadata.get("source", "Statistics Norway"),
            "variables": [],
            "aggregation_options": {},
            "data_coverage": {},
            "query_suggestions": [],
            "CRITICAL_API_DIMENSION_MAPPING": {
                "statistikkvariabel": "ContentsCode",
                "næring (SN2007)": "NACE2007", 
                "næring": "NACE2007",
                "region": "Region",
                "år": "Tid",
                "kvartal": "Tid",
                "måned": "Tid",
                "kjønn": "Kjonn",
                "alder": "Alder"
            },
            "api_dimension_warning": "⚠️ CRITICAL: NEVER use display names in API calls! Use the CRITICAL_API_DIMENSION_MAPPING above to translate dimension names before calling discover_dimension_values or get_filtered_data."
        }
        
        # Analyze dimensions from the metadata
        dimensions = metadata.get("dimension", {})
        
        for dim_name, dim_data in dimensions.items():
            # Get basic dimension info
            category = dim_data.get("category", {})
            codes = list(category.get("index", []))
            labels = category.get("label", {})
            
            var_info = {
                "code": dim_name,
                "label": dim_name,
                "type": "dimension",
                "total_values": len(codes),
                "sample_values": codes[:5] if codes else [],
                "sample_labels": [labels.get(code, code) for code in codes[:5]] if codes else [],
                "has_aggregation": False,
                "aggregation_options": []
            }
            
            # Check for aggregation options (code lists)
            extension = dim_data.get("extension", {})
            code_lists = extension.get("codeLists", [])  # This is a LIST, not a dict
            
            if code_lists:
                var_info["has_aggregation"] = True
                aggregation_info = {
                    "valuesets": [],
                    "aggregations": []
                }
                
                # Process the codeLists array directly
                for code_list_item in code_lists:
                    if isinstance(code_list_item, dict):
                        item_type = code_list_item.get("type", "").lower()
                        item_info = {
                            "id": code_list_item.get("id", ""),
                            "label": code_list_item.get("label", ""),
                            "type": item_type
                        }
                        
                        if item_type == "valueset":
                            item_info["usage"] = "Use with codelist parameter for predefined selections"
                            aggregation_info["valuesets"].append(item_info)
                            var_info["aggregation_options"].append(f"Valueset: {item_info['id']} - {item_info['label']}")
                        elif item_type == "aggregation":
                            item_info["usage"] = "Use with codelist + outputValues=aggregated/single"
                            aggregation_info["aggregations"].append(item_info)
                            var_info["aggregation_options"].append(f"Aggregation: {item_info['id']} - {item_info['label']}")
                
                analysis["aggregation_options"][dim_name] = aggregation_info
            
            analysis["variables"].append(var_info)
        
        analysis["data_coverage"]["total_data_combinations"] = f"{len(dimensions)} dimensions"
        analysis["data_coverage"]["exceeds_api_limit"] = len(dimensions) > 3
        
        # Generate enhanced suggestions based on discovered dimensions and aggregations
        suggestions = []
        
        # Find time-related dimensions
        time_dims = [dim for dim in dimensions.keys() if any(t in dim.lower() for t in ['tid', 'år', 'year', 'time', 'kvartal', 'måned'])]
        if time_dims:
            suggestions.append({
                "type": "recent_data", 
                "description": f"Get recent data using {time_dims[0]}",
                "example": f"Use time_selection='top(5)' or specific year like '2025'"
            })
        
        # Find region-related dimensions with aggregation suggestions
        region_dims = [dim for dim in dimensions.keys() if any(r in dim.lower() for r in ['region', 'kommune', 'fylke', 'geo'])]
        if region_dims:
            region_dim = region_dims[0]
            if region_dim in analysis["aggregation_options"]:
                agg_options = analysis["aggregation_options"][region_dim]
                if agg_options["aggregations"]:
                    for agg in agg_options["aggregations"]:
                        if "fylker" in agg["id"].lower():
                            suggestions.append({
                                "type": "county_aggregation",
                                "description": f"Get county-level data using {agg['id']}",
                                "example": f"Use code_lists={{'{region_dim}': '{agg['id']}'}} with outputValues={{'{region_dim}': 'single'}}"
                            })
                        elif "kommun" in agg["id"].lower():
                            suggestions.append({
                                "type": "municipality_aggregation", 
                                "description": f"Get municipality data using {agg['id']}",
                                "example": f"Use code_lists={{'{region_dim}': '{agg['id']}'}} with outputValues={{'{region_dim}': 'aggregated'}}"
                            })
            else:
                suggestions.append({
                    "type": "geographic_filter",
                    "description": f"Filter by location using {region_dim}",
                    "example": f"Use discover_dimension_values to find codes for {region_dim}"
                })
        
        # Generic dimension exploration
        if dimensions:
            suggestions.append({
                "type": "explore_dimensions",
                "description": "Explore available values for any dimension",
                "example": f"Use discover_dimension_values on dimensions like: {', '.join(list(dimensions.keys())[:3])}"
            })
        
        analysis["query_suggestions"] = suggestions
        
        return analysis
    
    except Exception as e:
        return {"error": f"Structure analysis failed: {str(e)}", "table_id": table_id}

@mcp.tool()
async def get_filtered_data(
    table_id: str = Field(description="SSB table identifier"),
    filters: dict = Field(description="REQUIRED: Dimension filters as dict, e.g. {'Region': '3806', 'ContentsCode': 'Folkemengde', 'Tid': '2023'}. Use codes from discover_dimension_values."),
    language: str = Field(default="no", pattern="^(no|en)$", description="Language preference"),
    output_format: str = Field(default="json-stat2", description="Output format"),
    max_data_points: int = Field(default=100, le=1000, description="Maximum data points to return"),
    time_selection: str = Field(default="", description="Advanced time selection: 'top(5)', 'from(2020)', 'range(2020,2023)', '2020*' for wildcards"),
    code_lists: dict = Field(default={}, description="Specify code lists: {'Region': 'vs_Fylker'} for counties, {'Region': 'vs_Kommun'} for municipalities"),
    output_values: dict = Field(default={}, description="Aggregation control: {'Region': 'aggregated'} or {'Region': 'single'}")
) -> dict:
    """
    Get filtered statistical data with advanced SSB API features.
    
    ADVANCED FILTERING CAPABILITIES:
    - Time selections: top(n), from(year), range(start,end), wildcards (2020*)
    - Code lists: vs_Fylker (counties), vs_Kommun (municipalities), agg_* (groupings)
    - Output values: 'aggregated' for sums, 'single' for individual values
    - Wildcards: ?? for 2-digit codes, * for all values
    
    Example: filters={'Region': 'K-3103', 'Tid': 'top(5)'}, code_lists={'Region': 'agg_KommSummer'}
    """
    
    try:
        # Validate that filters is provided and is a dict
        if not filters:
            return {
                "error": "filters parameter is required and cannot be empty",
                "suggestion": "Use discover_dimension_values to find correct codes, then provide filters like {'Region': '3806', 'Tid': '2023'}",
                "table_id": table_id
            }
        
        if not isinstance(filters, dict):
            return {
                "error": f"filters must be a dict, got {type(filters)}",
                "suggestion": "Use format: filters={'DimensionName': 'code'}",
                "table_id": table_id
            }
        
        base_url = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        data_url = f"{base_url}/{table_id}/data"
        
        # Build GET request with advanced SSB API features (GET method works better for advanced features)
        query_params = {
            "lang": language,
            "outputformat": output_format
        }
        
        # Add code list specifications (new feature)
        for dimension, code_list in code_lists.items():
            query_params[f"codelist[{dimension}]"] = code_list
            print(f"DEBUG: Using code list {dimension}={code_list}")
        
        # Add output value specifications (new feature)
        for dimension, output_type in output_values.items():
            query_params[f"outputValues[{dimension}]"] = output_type
            print(f"DEBUG: Using output values {dimension}={output_type}")
        
        # Handle filters with advanced syntax
        for dimension, values in filters.items():
            if isinstance(values, str):
                value_list = [values] if ',' not in values else values.split(',')
            else:
                value_list = values if isinstance(values, list) else [str(values)]
            
            query_params[f"valueCodes[{dimension}]"] = ",".join(value_list)
            print(f"DEBUG: Added filter {dimension}={query_params[f'valueCodes[{dimension}]']}")
        
        # Handle time_selection with advanced syntax
        if time_selection:
            query_params["valueCodes[Tid]"] = time_selection
            print(f"DEBUG: Using advanced time selection: {time_selection}")
        
        print(f"DEBUG: Using GET method with params: {query_params}")
        
        # No hardcoded mappings - let the agent discover codes dynamically
        
        print(f"DEBUG: Using GET method with advanced parameters")
        
        # Use GET method with advanced parameters
        data = await robust_api_call(data_url, query_params)
        if not data:
            return {"error": "Failed to fetch filtered data", "table_id": table_id}
        
        print(f"DEBUG: SSB API returned data with keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        
        # Check if we got an error response
        if isinstance(data, dict) and "error" in data:
            print(f"DEBUG: SSB API returned error: {data}")
        elif not data:
            print(f"DEBUG: SSB API returned empty/None data")
        elif isinstance(data, dict):
            print(f"DEBUG: SSB API success - processing {len(data.get('value', []))} values")
        
        if isinstance(data, dict) and "error" in data:
            return {
                "table_id": table_id,
                "error": f"Filtered data retrieval failed: {data['error']}",
                "details": data.get("details", {}),
                "filters_attempted": filters,
                "suggestion": data.get("suggestion", "Try simpler filters or check dimension codes")
            }
        
        # Process JSON-stat2 data for easier consumption
        result = {
            "table_id": table_id,
            "title": data.get("label", f"Table {table_id}"),
            "source": data.get("source", "Statistics Norway"),
            "updated": data.get("updated", ""),
            "filters_applied": filters,
            "dimensions": list(data.get("dimension", {}).keys()),
            "total_data_points": len([v for v in data.get("value", []) if v is not None]),
            "formatted_data": []
        }
        
        if 'dimension' in data and 'value' in data:
            dimensions = data["dimension"]
            values = data["value"]
            
            # Create structured data representation
            non_null_count = 0
            for i, value in enumerate(values):
                if value is not None and non_null_count < max_data_points:
                    data_point = {"value": value, "index": i}
                    
                    # Add dimension labels
                    for dim_name, dim_data in dimensions.items():
                        if 'category' in dim_data and 'label' in dim_data['category']:
                            labels = list(dim_data['category']['label'].values())
                            size = len(labels)
                            if size > 0:
                                data_point[dim_name] = labels[i % size]
                    
                    result["formatted_data"].append(data_point)
                    non_null_count += 1
            
            result["returned_data_points"] = non_null_count
            
            # Add summary statistics if numeric data
            numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
            if numeric_values:
                result["summary_stats"] = {
                    "count": len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "average": sum(numeric_values) / len(numeric_values)
                }
        
        return result
    
    except Exception as e:
        return {
            "table_id": table_id,
            "error": f"Filtered data retrieval failed: {str(e)}",
            "filters_applied": filters
        }

@mcp.tool()
async def search_region_codes(
    region_name: str = Field(description="Name of the region/municipality to search for"),
    language: str = Field(default="no", pattern="^(no|en)$", description="Language for the search")
) -> dict:
    """
    Search for current region codes by querying SSB table metadata.
    Uses known population tables to find region codes dynamically.
    """
    
    try:
        print(f"DEBUG: Searching for region codes for '{region_name}'")
        
        # First search for tables that might have region data
        search_url = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        search_params = {
            "query": f"title:region OR title:kommune OR title:fylke NOT \"closed series\"",
            "lang": language,
            "pageSize": 5
        }
        
        search_results = await robust_api_call(search_url, search_params)
        if not search_results or 'tables' not in search_results:
            return {
                "region_name": region_name,
                "error": "Could not find tables with region data",
                "suggestion": "Try using discover_dimension_values on a specific table"
            }
        
        tables_to_check = [table['id'] for table in search_results['tables'][:3]]
        
        for table_id in tables_to_check:
            print(f"DEBUG: Checking table {table_id} for region codes")
            
            # Get detailed metadata from the /metadata endpoint
            meta_url = f"https://data.ssb.no/api/pxwebapi/v2-beta/tables/{table_id}/metadata"
            meta_params = {"lang": language}
            
            metadata = await robust_api_call(meta_url, meta_params)
            if not metadata or 'dimension' not in metadata:
                print(f"DEBUG: No dimension data in table {table_id}")
                continue
            
            # Look for Region dimension
            dimensions = metadata.get('dimension', {})
            if 'Region' not in dimensions:
                print(f"DEBUG: No Region dimension in table {table_id}")
                continue
            
            region_dim = dimensions['Region']
            region_values = region_dim.get('category', {}).get('label', {})
            
            print(f"DEBUG: Found {len(region_values)} regions in table {table_id}")
            
            # Search for matching region names (case-insensitive)
            matching_codes = []
            region_name_lower = region_name.lower()
            
            for code, label in region_values.items():
                if region_name_lower in label.lower():
                    matching_codes.append({
                        "code": code,
                        "label": label,
                        "table_id": table_id
                    })
            
            if matching_codes:
                print(f"DEBUG: Found {len(matching_codes)} matches in table {table_id}")
                return {
                    "region_name": region_name,
                    "codes_found": matching_codes,
                    "total_found": len(matching_codes),
                    "suggestion": f"Use code(s) in Region filter: {[c['code'] for c in matching_codes]}",
                    "source_table": table_id
                }
        
        # If no matches found in any table
        return {
            "region_name": region_name,
            "codes_found": [],
            "error": f"No matching region codes found for '{region_name}' in tables {known_tables}",
            "suggestion": "Try a different spelling, check if the region exists, or use a broader search term"
        }
        
    except Exception as e:
        print(f"DEBUG: Error in search_region_codes: {str(e)}")
        return {
            "region_name": region_name,
            "codes_found": [],
            "error": f"Failed to search for region codes: {str(e)}",
            "suggestion": "Try using the region name directly or check table structure manually"
        }

@mcp.tool()
async def discover_code_lists(
    table_id: str = Field(description="SSB table identifier"),
    dimension_name: str = Field(description="Dimension to explore (e.g., 'Region', 'ContentsCode')"),
    language: str = Field(default="no", pattern="^(no|en)$", description="Language preference")
) -> dict:
    """
    Discover available code lists (valuesets and groupings) for a dimension.
    
    Code lists include:
    - vs_*: Valuesets (predefined selections like vs_Fylker for counties)
    - agg_*: Aggregation groupings (e.g., agg_Fylker2024 for current counties)
    
    This helps choose the right administrative level and groupings for filtering.
    """
    
    try:
        print(f"DEBUG: Discovering code lists for {dimension_name} in table {table_id}")
        
        # Get detailed metadata to find code lists
        meta_url = f"https://data.ssb.no/api/pxwebapi/v2-beta/tables/{table_id}/metadata"
        params = {"lang": language}
        
        metadata = await robust_api_call(meta_url, params)
        if not metadata or 'dimension' not in metadata:
            return {
                "table_id": table_id,
                "dimension_name": dimension_name,
                "error": "Could not retrieve table metadata",
                "suggestion": "Check table ID and try again"
            }
        
        dimensions = metadata.get('dimension', {})
        if dimension_name not in dimensions:
            available_dims = list(dimensions.keys())
            return {
                "table_id": table_id,
                "dimension_name": dimension_name,
                "error": f"Dimension '{dimension_name}' not found",
                "available_dimensions": available_dims,
                "suggestion": f"Use one of: {available_dims}"
            }
        
        dim_data = dimensions[dimension_name]
        extension = dim_data.get('extension', {})
        code_lists = extension.get('codeLists', [])
        
        result = {
            "table_id": table_id,
            "dimension_name": dimension_name,
            "code_lists": {},
            "recommendations": []
        }
        
        # Process the codeLists array directly
        for code_list_item in code_lists:
            if isinstance(code_list_item, dict):
                item_id = code_list_item.get('id', '')
                item_label = code_list_item.get('label', '')
                item_type = code_list_item.get('type', '').lower()
                
                result["code_lists"][item_id] = {
                    "type": item_type,
                    "label": item_label,
                    "description": f"{item_type.title()}: {item_label}",
                    "usage_example": f"code_lists={{'{dimension_name}': '{item_id}'}}" + (
                        f" with outputValues={{'{dimension_name}': 'aggregated' or 'single'}}" if item_type == "aggregation" else ""
                    )
                }
                
                # Add specific recommendations based on type and content
                if item_type == "valueset":
                    if 'fylker' in item_id.lower() or 'county' in item_label.lower():
                        result["recommendations"].append(f"Use '{item_id}' for county-level analysis")
                    elif 'kommun' in item_id.lower() or 'municipal' in item_label.lower():
                        result["recommendations"].append(f"Use '{item_id}' for municipality-level analysis")
                    elif 'landet' in item_id.lower():
                        result["recommendations"].append(f"Use '{item_id}' for national-level data")
                
                elif item_type == "aggregation":
                    if 'fylker' in item_id.lower():
                        result["recommendations"].append(f"Use '{item_id}' with outputValues[{dimension_name}]=single for current county structure")
                        result["recommendations"].append(f"Example: Get all current counties with filters={{'Region': '*', 'Tid': 'top(1)'}}, code_lists={{'{dimension_name}': '{item_id}'}}, output_values={{'{dimension_name}': 'single'}}")
                    elif 'komm' in item_id.lower():
                        result["recommendations"].append(f"Use '{item_id}' with outputValues[{dimension_name}]=aggregated for merged municipalities")
                        result["recommendations"].append(f"Example: Get aggregated municipality data with code_lists={{'{dimension_name}': '{item_id}'}}, output_values={{'{dimension_name}': 'aggregated'}}")
        
        if not result["code_lists"]:
            result["message"] = f"No special code lists found for {dimension_name}. Use standard dimension values."
            result["fallback_suggestion"] = f"Use discover_dimension_values('{table_id}', '{dimension_name}') to see all available values"
        
        return result
        
    except Exception as e:
        print(f"DEBUG: Error discovering code lists: {str(e)}")
        return {
            "table_id": table_id,
            "dimension_name": dimension_name,
            "error": f"Failed to discover code lists: {str(e)}",
            "suggestion": "Try using standard dimension values with discover_dimension_values"
        }

@mcp.tool()
async def diagnose_table_requirements(
    table_id: str = Field(description="SSB table identifier that failed"),
    error_message: str = Field(description="Error message received from SSB API"),
    attempted_filters: dict = Field(description="Filters that were attempted"),
    language: str = Field(default="no", pattern="^(no|en)$", description="Language preference")
) -> dict:
    """
    Diagnose why a table query failed and suggest corrections.
    
    This tool helps the agent understand:
    - Which dimensions are mandatory vs optional
    - What the correct dimension names are (not assumptions)
    - What codes are actually available for each dimension
    - How to build a complete filter set for any table
    
    Use this when get_filtered_data fails to learn the table's requirements.
    """
    
    try:
        print(f"DEBUG: Diagnosing table {table_id} failure: {error_message}")
        
        # Get complete table metadata
        meta_url = f"https://data.ssb.no/api/pxwebapi/v2-beta/tables/{table_id}/metadata"
        params = {"lang": language}
        
        metadata = await robust_api_call(meta_url, params)
        if not metadata:
            return {
                "table_id": table_id,
                "error": "Could not retrieve table metadata for diagnosis",
                "suggestion": "Check table ID and try again"
            }
        
        dimensions = metadata.get('dimension', {})
        diagnosis = {
            "table_id": table_id,
            "error_analysis": error_message,
            "attempted_filters": attempted_filters,
            "all_dimensions": {},
            "missing_mandatory": [],
            "suggestions": []
        }
        
        # Analyze each dimension
        for dim_name, dim_data in dimensions.items():
            category = dim_data.get('category', {})
            codes = category.get('index', [])
            labels = category.get('label', [])
            
            # Check if this dimension was attempted
            was_attempted = dim_name in attempted_filters
            
            diagnosis["all_dimensions"][dim_name] = {
                "attempted": was_attempted,
                "total_values": len(codes),
                "sample_codes": codes[:5] if codes else [],
                "sample_labels": labels[:5] if len(labels) >= 5 else labels,
                "has_total_code": any(code in ['0', '00-99', '*', 'I alt'] for code in codes[:10])
            }
            
            # If not attempted and error mentions missing selection, it's likely mandatory
            if not was_attempted and "missing selection" in error_message.lower():
                diagnosis["missing_mandatory"].append(dim_name)
        
        # Generate specific suggestions
        if diagnosis["missing_mandatory"]:
            diagnosis["suggestions"].append(
                f"Add these mandatory dimensions to filters: {diagnosis['missing_mandatory']}"
            )
            
            for dim in diagnosis["missing_mandatory"]:
                dim_info = diagnosis["all_dimensions"][dim]
                if dim_info["has_total_code"]:
                    diagnosis["suggestions"].append(
                        f"For {dim}, try total codes like: {[c for c in dim_info['sample_codes'] if c in ['0', '00-99', '*', 'I alt']]}"
                    )
                else:
                    diagnosis["suggestions"].append(
                        f"For {dim}, use discover_dimension_values to find appropriate codes"
                    )
        
        diagnosis["suggestions"].append(
            "Use the EXACT dimension names from this analysis - no assumptions"
        )
        
        return diagnosis
        
    except Exception as e:
        print(f"DEBUG: Error in diagnosis: {str(e)}")
        return {
            "table_id": table_id,
            "error": f"Diagnosis failed: {str(e)}",
            "suggestion": "Try analyze_table_structure and discover_dimension_values manually"
        }

@mcp.tool()
async def web_search_ssb_info(
    search_query: str = Field(description="Search query for SSB-related information"),
    language: str = Field(default="no", pattern="^(no|en)$", description="Language preference")
) -> dict:
    """
    Search the web for current SSB information when API metadata is insufficient.
    Useful for finding current municipality codes, table IDs, or understanding data availability.
    """
    
    try:
        # Construct search query focused on SSB
        ssb_query = f"site:ssb.no {search_query} statistikk kommune kode"
        if language == "en":
            ssb_query = f"site:ssb.no {search_query} statistics municipality code"
        
        print(f"DEBUG: Web searching for SSB info: '{ssb_query}'")
        
        # Use a simple web search (you could integrate with a proper search API)
        search_url = "https://www.google.com/search"
        search_params = {
            "q": ssb_query,
            "num": 5,
            "hl": language
        }
        
        # For now, return a structured suggestion since we can't do actual web scraping
        # In a production system, you'd integrate with a proper search API
        return {
            "search_query": search_query,
            "suggestion": f"Search SSB.no manually for: '{search_query}'",
            "recommended_actions": [
                "Check SSB.no for current municipality codes",
                "Look for updated table documentation",
                "Verify data availability for the specific region",
                "Try alternative table IDs if current one has issues"
            ],
            "fallback_strategy": "Use analyze_table_structure to explore available regions in different tables"
        }
        
    except Exception as e:
        return {
            "search_query": search_query,
            "error": f"Web search failed: {str(e)}",
            "suggestion": "Try using SSB API tools to explore available data"
        }

@mcp.tool()
async def discover_dimension_values(
    table_id: str = Field(description="SSB table identifier"),
    dimension_name: str = Field(description="Name of dimension to explore (e.g., 'Region', 'ContentsCode', 'Tid')"),
    search_term: str = Field(default="", description="Optional search term to filter values"),
    language: str = Field(default="no", pattern="^(no|en)$", description="Language preference")
) -> dict:
    """
    Discover actual available values for any dimension in a table.
    This is the key tool for learning what values are actually available.
    """
    
    try:
        print(f"DEBUG: Discovering values for dimension '{dimension_name}' in table {table_id}")
        
        # Get detailed metadata from the /metadata endpoint
        meta_url = f"https://data.ssb.no/api/pxwebapi/v2-beta/tables/{table_id}/metadata"
        meta_params = {"lang": language}
        
        metadata = await robust_api_call(meta_url, meta_params)
        if not metadata or 'dimension' not in metadata:
            return {
                "table_id": table_id,
                "dimension_name": dimension_name,
                "error": "Could not retrieve table metadata",
                "suggestion": "Check if table exists and dimension name is correct"
            }
        
        # Look for the specific dimension
        dimensions = metadata.get('dimension', {})
        if dimension_name not in dimensions:
            available_dims = list(dimensions.keys())
            # Provide clear learning opportunity for the agent
            return {
                "table_id": table_id,
                "dimension_name": dimension_name,
                "error": f"Dimension '{dimension_name}' not found",
                "available_dimensions": available_dims,
                "suggestion": f"Use one of: {available_dims}",
                "pattern_hint": f"Notice: '{dimension_name}' → try '{available_dims[0]}' (common pattern: lowercase → CamelCase or Norwegian → English)"
            }
        
        dim_data = dimensions[dimension_name]
        all_values = dim_data.get('category', {}).get('label', {})
        
        print(f"DEBUG: Found {len(all_values)} values in dimension {dimension_name}")
        
        # Filter values if search term provided
        if search_term:
            search_lower = search_term.lower()
            filtered_values = {}
            for code, label in all_values.items():
                if search_lower in label.lower() or search_lower in code.lower():
                    filtered_values[code] = label
            values_to_show = filtered_values
        else:
            values_to_show = all_values
        
        # Convert to list format for easier consumption
        value_list = [
            {"code": code, "label": label}
            for code, label in values_to_show.items()
        ]
        
        # Limit output for readability
        if len(value_list) > 20:
            shown_values = value_list[:20]
            truncated = len(value_list) - 20
        else:
            shown_values = value_list
            truncated = 0
        
        return {
            "table_id": table_id,
            "dimension_name": dimension_name,
            "search_term": search_term,
            "total_values": len(all_values),
            "matching_values": len(value_list),
            "values": shown_values,
            "truncated_count": truncated,
            "suggestion": f"Use the 'code' values in filters for {dimension_name}"
        }
        
    except Exception as e:
        print(f"DEBUG: Error in discover_dimension_values: {str(e)}")
        return {
            "table_id": table_id,
            "dimension_name": dimension_name,
            "error": f"Failed to discover dimension values: {str(e)}",
            "suggestion": "Check table ID and dimension name"
        }

if __name__ == "__main__":
    mcp.run()