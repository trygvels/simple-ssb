#!/usr/bin/env python3
"""
SSB Data Analysis MCP Server - Intelligent Statistical Data Discovery & Analysis

AVAILABLE TOOLS FOR DATA ANALYSIS:
==================================

🔍 search_tables(query: str) -> dict
   Purpose: Discovery and selection of statistical tables from SSB's database
   Agent Use: Find relevant tables by search terms, get ranked results with metadata
   Data Analysis: Essential first step - identifies available datasets for analysis
   Output: Table IDs, titles, time periods, variable counts, relevance scores

📊 get_table_info(table_id: str, include_structure: bool = True) -> dict
   Purpose: Complete table structure analysis with API dimension mapping
   Agent Use: Understand table contents, dimensions, time coverage, data availability
   Data Analysis: Critical for query planning - reveals all analysis dimensions
   Output: Variable details, API names, sample values, aggregation options, workflow guidance

🎯 discover_dimension_values(table_id: str, dimension_name: str, search_term: str = "", include_code_lists: bool = True) -> dict
   Purpose: Explore dimension values, administrative levels, and filtering codes
   Agent Use: Get all available codes for filtering, find regional/categorical breakdowns
   Data Analysis: Enables precise data segmentation and comparative analysis
   Output: All dimension codes with labels, administrative groupings, usage guidance

📈 get_filtered_data(table_id: str, filters: dict, time_selection: str = "", code_lists: dict = {}) -> dict
   Purpose: Extract specific statistical data with intelligent error handling
   Agent Use: Retrieve actual data points for analysis with proper filtering
   Data Analysis: Core data extraction for statistical analysis and visualization
   Output: Structured data with dimensions, summary statistics, diagnostic guidance


DATA ANALYSIS CAPABILITIES:
===========================
✅ Cross-Domain Analysis: Works identically for employment, demographics, housing, healthcare, etc.
✅ Time Series Analysis: Automated time dimension discovery and period selection
✅ Regional Analysis: Municipal, county, and national level data with administrative codes
✅ Comparative Analysis: Multi-dimensional breakdowns for statistical comparisons
✅ Trend Analysis: Historical data spanning decades with consistent methodology
✅ Error-Driven Learning: Intelligent error handling teaches proper API usage patterns

INTELLIGENCE FEATURES:
=====================
🧠 Self-Learning: Discovers SSB API patterns through intelligent error analysis
🎯 Domain-Agnostic: No hardcoded assumptions - adapts to any statistical domain
🔄 Workflow Guidance: Each tool suggests logical next steps for analysis
🛠️ API Mastery: Automatic translation between Norwegian terms and API requirements
📊 Quality Assurance: Data validation, summary statistics, and diagnostic information

This MCP server transforms SSB's complex statistical API into an intelligent, 
agent-friendly interface optimized for comprehensive data analysis workflows.
"""

import asyncio
import httpx
from typing import Optional
from fastmcp import FastMCP
from pydantic import Field
from datetime import datetime, timedelta

# Create FastMCP server
mcp = FastMCP("Clean SSB Discovery")

# Rate limiter 
class RateLimiter:
    def __init__(self, max_calls=25, time_window=600):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    async def acquire(self):
        now = datetime.now()
        self.calls = [call for call in self.calls if call > now - timedelta(seconds=self.time_window)]

        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(min(sleep_time, 2.0))

        self.calls.append(now)

# Global instances
rate_limiter = RateLimiter()


def create_error_response(tool_name: str, table_id: str = None, error_msg: str = "", suggestion: str = "", **extras) -> dict:
    """Create standardized error response with agent guidance."""
    response = {
        "error": error_msg,
        "suggestion": suggestion,
        "agent_guidance": {
            "recommended_action": "retry_with_corrections",
            "tool_name": tool_name
        }
    }
    
    if table_id:
        response["table_id"] = table_id
        
    response.update(extras)
    return response

def enhance_variable_info(variables: list) -> list:
    """Enhance variable information for agent use."""
    enhanced = []
    
    for var in variables:
        enhanced.append({
            "display_name": var,
            "api_name": var,  # Use actual API name directly
            "is_mapped": False,  # No mapping needed - use real names
            "pattern_hint": f"Use '{var}' in API calls"
        })
    
    return enhanced

def add_agent_guidance(tool_name: str, result: dict, **context) -> dict:
    """Add agent-specific guidance to tool results."""
    
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
    
    base_guidance = guidance_map.get(tool_name, {})
    
    result["agent_guidance"] = {
        "tool_name": tool_name,
        "status": "success" if "error" not in result else "error",
        **base_guidance,
        **context
    }
    
    return result

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
            if e.response.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return {
                    "error": f"HTTP {e.response.status_code} error from SSB API",
                    "suggestion": "Check API parameters and table availability"
                }

        except (httpx.RequestError, asyncio.TimeoutError) as e:
            print(f"DEBUG: Request/Timeout error: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

    return None

@mcp.tool()
async def search_tables(
    query: str = Field(description="Search query for SSB statistical tables")
) -> dict:
    """
    Advanced search for SSB statistical tables using official API search syntax.
    Generates multiple query variations and scores results for relevance.
    """

    try:
        print(f"DEBUG: Searching for '{query}'")
        
        base_url = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        language = "no"  # Fixed Norwegian language
        max_results = 10  # Fixed default
        
        # Generate simple query variations
        query_variations = [
            query,
            f'title:{query} NOT "closed series"',
            f'title:{query} AND title:(K) NOT "closed series"'  # Municipal scope
        ]
        
        all_tables = {}
        
        # Try query variations
        for search_query in query_variations[:2]:  # Limit to 2 for speed
            params = {
                "query": search_query,
                "lang": language,
                "pageSize": max_results
            }

            data = await robust_api_call(base_url, params)
            
            if data and 'tables' in data:
                for table in data.get('tables', []):
                    table_id = table.get('id')
                    if table_id and table_id not in all_tables:
                        all_tables[table_id] = table

        tables = list(all_tables.values())
        
        # Simple scoring
        query_words = query.lower().split()
        scored_tables = []

        for table in tables:
            label = table.get('label', '').lower()
            score = 1
            
            for word in query_words:
                if len(word) > 2 and word in label:
                    score += 10

            # Boost for recent updates
            if '2024' in table.get('updated', '') or '2025' in table.get('updated', ''):
                score += 5

            scored_tables.append({
                'id': table.get('id'),
                'title': table.get('label'),
                'description': table.get('description', ''),
                'updated': table.get('updated', ''),
                'score': score,
                'time_period': f"{table.get('firstPeriod', '')} - {table.get('lastPeriod', '')}",
                'variables': len(table.get('variableNames', [])),
                'subject_area': table.get('subjectCode', '')
            })

        scored_tables.sort(key=lambda x: x['score'], reverse=True)

        result = {
            "query": query,
            "language": language,
            "tables": scored_tables[:max_results],
            "total_found": len(scored_tables),
            "search_tips": [
                "Try Norwegian keywords for better results",
                "Use specific terms rather than general ones"
            ]
        }

        return add_agent_guidance("search_tables", result)

    except Exception as e:
        return {"error": f"Search failed: {str(e)}", "tables": [], "total_found": 0}

@mcp.tool()
async def get_table_info(
    table_id: str = Field(description="SSB table identifier"),
    include_structure: bool = Field(default=True, description="Include detailed structure analysis")
) -> dict:
    """
    Get complete table information combining metadata and detailed structure analysis.
    This consolidated tool provides both basic metadata and detailed dimension information.
    """

    try:
        print(f"DEBUG: Getting comprehensive info for table {table_id}")

        base_url = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        info_url = f"{base_url}/{table_id}"
        metadata_url = f"{base_url}/{table_id}/metadata"
        language = "no"  # Fixed Norwegian language
        params = {"lang": language}

        # Get basic table info first
        basic_data = await robust_api_call(info_url, params)
        
        if not basic_data:
            return create_error_response(
                "get_table_info", 
                table_id=table_id,
                error_msg="Failed to retrieve table information",
                suggestion="Check network connection and table ID"
            )
        
        if not basic_data.get("label"):
            return create_error_response(
                "get_table_info",
                table_id=table_id, 
                error_msg="Table not found or unavailable",
                suggestion="Check table ID or use search_tables"
            )

        # Build basic information
        variables = basic_data.get("variableNames", [])
        enhanced_variables = enhance_variable_info(variables)
        
        comprehensive_info = {
            "table_id": table_id,
            "title": basic_data.get("label", ""),
            "description": basic_data.get("description", ""),
            "first_period": basic_data.get("firstPeriod", ""),
            "last_period": basic_data.get("lastPeriod", ""),
            "last_updated": basic_data.get("updated", ""),
            "source": "Statistics Norway (SSB)",
            "variables": enhanced_variables,
            "total_variables": len(variables),
            "time_span": f"{basic_data.get('firstPeriod', 'Unknown')} - {basic_data.get('lastPeriod', 'Unknown')}",
            "data_availability": {
                "has_recent_data": "2024" in str(basic_data.get("lastPeriod", "")) or "2025" in str(basic_data.get("lastPeriod", "")),
                "time_coverage": basic_data.get("lastPeriod", "Unknown"),
                "frequency": "quarterly" if "K" in str(basic_data.get("lastPeriod", "")) else "monthly" if "M" in str(basic_data.get("lastPeriod", "")) else "annual"
            }
        }

        # Get detailed structure if requested
        if include_structure:
            metadata = await robust_api_call(metadata_url, params)
            if metadata and isinstance(metadata, dict):
                dimensions = metadata.get("dimension", {})
                detailed_variables = []
                aggregation_options = {}
                
                for dim_name, dim_data in dimensions.items():
                    category = dim_data.get("category", {})
                    codes = list(category.get("index", []))
                    labels = category.get("label", {})
                    
                    # Find corresponding enhanced variable info
                    enhanced_var = next((v for v in enhanced_variables if v["display_name"] == dim_name), None)
                    api_name = enhanced_var["api_name"] if enhanced_var else dim_name
                    
                    detailed_var = {
                        "display_name": dim_name,
                        "api_name": api_name,
                        "is_mapped": enhanced_var["is_mapped"] if enhanced_var else False,
                        "pattern_hint": enhanced_var["pattern_hint"] if enhanced_var else None,
                        "type": "dimension",
                        "total_values": len(codes),
                        "sample_values": codes[:5] if codes else [],
                        "sample_labels": [labels.get(code, code) for code in codes[:5]] if codes else []
                    }
                    
                    # Check for aggregation options
                    extension = dim_data.get("extension", {})
                    code_lists = extension.get("codeLists", [])
                    
                    if code_lists:
                        agg_info = {"valuesets": [], "aggregations": []}
                        for code_list_item in code_lists:
                            if isinstance(code_list_item, dict):
                                item_type = code_list_item.get("type", "").lower()
                                item_info = {
                                    "id": code_list_item.get("id", ""),
                                    "label": code_list_item.get("label", ""),
                                    "type": item_type
                                }
                                
                                if item_type == "valueset":
                                    agg_info["valuesets"].append(item_info)
                                elif item_type == "aggregation":
                                    agg_info["aggregations"].append(item_info)
                        
                        if agg_info["valuesets"] or agg_info["aggregations"]:
                            aggregation_options[dim_name] = agg_info
                    
                    detailed_variables.append(detailed_var)
                
                comprehensive_info["variables"] = detailed_variables
                comprehensive_info["aggregation_options"] = aggregation_options
                comprehensive_info["structure_included"] = True
            else:
                comprehensive_info["structure_included"] = False
        else:
            comprehensive_info["structure_included"] = False

        # Add workflow guidance
        comprehensive_info["workflow_guidance"] = {
            "next_steps": [
                "Use discover_dimension_values to get specific codes for dimensions",
                "Use get_filtered_data for data retrieval with proper API dimension names"
            ],
            "recommended_tools": ["discover_dimension_values", "get_filtered_data"],
            "workflow_hint": "Complete table info ready - use api_name values in subsequent calls"
        }

        return add_agent_guidance("get_table_info", comprehensive_info)

    except Exception as e:
        print(f"DEBUG: Error getting comprehensive table info: {str(e)}")
        return create_error_response(
            "get_table_info",
            table_id=table_id,
            error_msg=f"Unexpected error: {str(e)}",
            suggestion="Try again or contact support"
        )

@mcp.tool()
async def discover_dimension_values(
    table_id: str = Field(description="SSB table identifier"),
    dimension_name: str = Field(description="Name of dimension to explore (e.g., 'Region', 'ContentsCode', 'Tid')"),
    search_term: str = Field(default="", description="Optional search term to filter values"),
    include_code_lists: bool = Field(default=True, description="Include available code lists and aggregations")
) -> dict:
    """
    Discover available values for any dimension, with optional code lists and aggregations.
    Consolidated functionality from discover_dimension_values + discover_code_lists + search_region_codes.
    """

    try:
        print(f"DEBUG: Discovering values for dimension '{dimension_name}' in table {table_id}")

        meta_url = f"https://data.ssb.no/api/pxwebapi/v2-beta/tables/{table_id}/metadata"
        language = "no"  # Fixed Norwegian language
        meta_params = {"lang": language}

        metadata = await robust_api_call(meta_url, meta_params)
        if not metadata or 'dimension' not in metadata:
            return create_error_response(
                "discover_dimension_values",
                table_id=table_id,
                error_msg="Could not retrieve table metadata",
                suggestion="Check table ID and dimension name"
            )

        dimensions = metadata.get('dimension', {})
        if dimension_name not in dimensions:
            available_dims = list(dimensions.keys())
            return create_error_response(
                "discover_dimension_values",
                table_id=table_id,
                error_msg=f"Dimension '{dimension_name}' not found",
                suggestion=f"Use one of: {available_dims}",
                available_dimensions=available_dims,
                pattern_hint=f"Notice: '{dimension_name}' → try '{available_dims[0]}' (common pattern: lowercase → CamelCase)"
            )

        dim_data = dimensions[dimension_name]
        all_values = dim_data.get('category', {}).get('label', {})

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

        # Convert to list format
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

        result = {
            "table_id": table_id,
            "dimension_name": dimension_name,
            "search_term": search_term,
            "total_values": len(all_values),
            "matching_values": len(value_list),
            "values": shown_values,
            "truncated_count": truncated,
            "usage_suggestion": f"Use the 'code' values in filters for {dimension_name}"
        }

        # Add code lists if requested
        if include_code_lists:
            extension = dim_data.get('extension', {})
            code_lists = extension.get('codeLists', [])
            
            result["code_lists"] = {}
            result["recommendations"] = []
            
            for code_list_item in code_lists:
                if isinstance(code_list_item, dict):
                    item_id = code_list_item.get('id', '')
                    item_label = code_list_item.get('label', '')
                    item_type = code_list_item.get('type', '').lower()

                    result["code_lists"][item_id] = {
                        "type": item_type,
                        "label": item_label,
                        "usage_example": f"code_lists={{'{dimension_name}': '{item_id}'}}"
                    }

                    # Add dynamic recommendations based on content analysis
                    if item_type == "valueset":
                        if 'fylker' in item_id.lower() or 'county' in item_label.lower():
                            result["recommendations"].append(f"Use '{item_id}' for regional grouping analysis")
                        elif 'kommun' in item_id.lower() or 'municipal' in item_label.lower():
                            result["recommendations"].append(f"Use '{item_id}' for local area analysis")
                        else:
                            result["recommendations"].append(f"Use '{item_id}' for {item_label.lower()} grouping")
                    elif item_type == "aggregation":
                        result["recommendations"].append(f"Use '{item_id}' for aggregated {item_label.lower()} data")

        return add_agent_guidance("discover_dimension_values", result)

    except Exception as e:
        print(f"DEBUG: Error in discover_dimension_values: {str(e)}")
        return create_error_response(
            "discover_dimension_values",
            table_id=table_id,
            error_msg=f"Failed to discover dimension values: {str(e)}",
            suggestion="Check table ID and dimension name"
        )

@mcp.tool()
async def get_filtered_data(
    table_id: str = Field(description="SSB table identifier"),
    filters: dict = Field(description="REQUIRED: Dimension filters as dict, e.g. {'Region': '3806', 'ContentsCode': 'Folkemengde', 'Tid': '2023'}"),
    time_selection: str = Field(default="", description="Advanced time selection: 'top(5)', 'from(2020)', '2020*' for wildcards"),
    code_lists: dict = Field(default={}, description="Code lists: {'Region': 'vs_Fylker'} for counties")
) -> dict:
    """
    Get filtered statistical data with enhanced error handling and validation.
    Includes diagnostic information when queries fail.
    """

    try:
        # Debug the filters parameter type
        print(f"DEBUG: filters type: {type(filters)}, value: {filters}")
        
        # Handle potential Field object from FastMCP
        if hasattr(filters, 'default') and hasattr(filters, 'description'):
            # This is a Field object, use its default value
            filters = filters.default if filters.default is not None else {}
        
        # Validate filters
        if not filters or not isinstance(filters, dict):
            return create_error_response(
                "get_filtered_data",
                table_id=table_id,
                error_msg="filters parameter is required and must be a dict",
                suggestion="Use discover_dimension_values to find correct codes, then provide filters like {'Region': '3806', 'Tid': '2023'}"
            )

        base_url = "https://data.ssb.no/api/pxwebapi/v2-beta/tables"
        data_url = f"{base_url}/{table_id}/data"
        language = "no"  # Fixed Norwegian language
        max_data_points = 100  # Fixed default

        # Build query parameters
        query_params = {
            "lang": language,
            "outputformat": "json-stat2"
        }

        # Add code lists (handle potential Field objects)
        if hasattr(code_lists, 'default'):
            code_lists = code_lists.default if code_lists.default is not None else {}
        
        if isinstance(code_lists, dict):
            for dimension, code_list in code_lists.items():
                query_params[f"codelist[{dimension}]"] = code_list

        # Handle filters (ensure filters is a dict)
        if isinstance(filters, dict):
            for dimension, values in filters.items():
                if isinstance(values, str):
                    value_list = [values] if ',' not in values else values.split(',')
                else:
                    value_list = values if isinstance(values, list) else [str(values)]

                query_params[f"valueCodes[{dimension}]"] = ",".join(value_list)

        # Handle time selection (handle potential Field objects)
        if hasattr(time_selection, 'default'):
            time_selection = time_selection.default if time_selection.default is not None else ""
        
        if time_selection and isinstance(time_selection, str):
            query_params["valueCodes[Tid]"] = time_selection

        print(f"DEBUG: Getting filtered data with params: {query_params}")

        data = await robust_api_call(data_url, query_params)
        if not data:
            return create_error_response(
                "get_filtered_data",
                table_id=table_id,
                error_msg="Failed to fetch filtered data",
                suggestion="Check filters and try again"
            )

        # Check for API errors
        if isinstance(data, dict) and "error" in data:
            # Enhanced error response with diagnostic info
            error_response = create_error_response(
                "get_filtered_data",
                table_id=table_id,
                error_msg=f"SSB API error: {data['error']}",
                suggestion="Check dimension names and codes using discover_dimension_values",
                filters_attempted=filters,
                diagnostic_hint="Use discover_dimension_values to verify dimension names and available codes"
            )
            
            # Add details if available
            if "details" in data:
                error_response["api_details"] = data["details"]
                
            return error_response

        # Process successful response
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

            # Add summary statistics
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
        return create_error_response(
            "get_filtered_data",
            table_id=table_id,
            error_msg=f"Unexpected error: {str(e)}",
            suggestion="Check parameters and try again",
            filters_applied=filters
        )


if __name__ == "__main__":
    mcp.run()