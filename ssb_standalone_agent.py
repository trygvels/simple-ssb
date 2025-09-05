#!/usr/bin/env python3
"""
SSB Standalone Agent - Complete Norwegian Statistical Data Analysis Agent

This script combines all MCP server functionality with the optimized agent
in a single standalone Python file. No MCP protocol needed - direct function calls.

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
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

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

# Configuration
model = "gpt-5"
console = Console()

# Setup logging - disable HTTP request logs for cleaner output
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=False, show_path=False, markup=True)]
)

# Specifically disable httpx logging which shows all HTTP requests
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

for handler in logging.root.handlers:
    if isinstance(handler, RichHandler):
        handler.setLevel(logging.WARNING)

# ============================================================================
# SSB API TOOLS - Direct Function Implementations
# ============================================================================

class RateLimiter:
    """Rate limiter for SSB API compliance (30 calls per 10 minutes)"""
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

# Global rate limiter
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

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                result = response.json()
                return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return {
                    "error": f"HTTP {e.response.status_code} error from SSB API",
                    "suggestion": "Check API parameters and table availability"
                }

        except (httpx.RequestError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

    return None

@function_tool
async def search_tables(query: str) -> dict:
    """
    Advanced search for SSB statistical tables using official API search syntax.
    Generates multiple query variations and scores results for relevance.
    
    Args:
        query: Search terms for finding SSB statistical tables (Norwegian keywords work best)
    
    Returns:
        dict: Search results with scored tables and metadata
    """
    try:
        
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

@function_tool
async def get_table_info(table_id: str, include_structure: bool = True) -> dict:
    """
    Get complete table information combining metadata and detailed structure analysis.
    This consolidated tool provides both basic metadata and detailed dimension information.
    
    Args:
        table_id: SSB table identifier (e.g., '07459', '13517')
        include_structure: Whether to include detailed structure analysis
    
    Returns:
        dict: Complete table metadata and structure information
    """
    try:

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
        return create_error_response(
            "get_table_info",
            table_id=table_id,
            error_msg=f"Unexpected error: {str(e)}",
            suggestion="Try again or contact support"
        )

@function_tool 
async def discover_dimension_values(table_id: str, dimension_name: str, search_term: str = "", include_code_lists: bool = True) -> dict:
    """
    Discover available values for any dimension, with optional code lists and aggregations.
    Consolidated functionality from discover_dimension_values + discover_code_lists + search_region_codes.
    
    Args:
        table_id: SSB table identifier
        dimension_name: Name of the dimension to explore (must match exact API name)
        search_term: Optional term to filter values
        include_code_lists: Whether to include code lists and aggregations
    
    Returns:
        dict: Available dimension values, code lists, and recommendations
    """
    try:

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
        return create_error_response(
            "discover_dimension_values",
            table_id=table_id,
            error_msg=f"Failed to discover dimension values: {str(e)}",
            suggestion="Check table ID and dimension name"
        )

@function_tool
async def get_filtered_data(table_id: str, filters_json: str, time_selection: str = "", code_lists_json: str = "") -> dict:
    """
    Get filtered statistical data with enhanced error handling and validation.
    Includes diagnostic information when queries fail.
    
    Args:
        table_id: SSB table identifier
        filters_json: JSON string of dimension filters (e.g., '{"Region": "0", "Tid": "2024"}')
        time_selection: Optional time selection (deprecated - use filters['Tid'] instead)
        code_lists_json: JSON string of code lists for aggregation (e.g., '{"NACE2007": "agg_NACE2007arb11"}')
    
    Returns:
        dict: Filtered data with formatted results and summary statistics
    """
    try:
        # Parse JSON strings to dictionaries
        filters = json.loads(filters_json) if filters_json else {}
        code_lists = json.loads(code_lists_json) if code_lists_json else None
        
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

        # Add code lists
        if code_lists and isinstance(code_lists, dict):
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

        # Handle time selection
        if time_selection and isinstance(time_selection, str):
            query_params["valueCodes[Tid]"] = time_selection


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

# ============================================================================
# STANDALONE AGENT IMPLEMENTATION
# ============================================================================

class SSBStandaloneAgent:
    """Standalone SSB Agent with direct function calls - no MCP needed."""
    
    def __init__(self):
        # Load environment variables
        simple_ssb_env = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
        if os.path.exists(simple_ssb_env):
            load_dotenv(simple_ssb_env, override=True)
        else:
            env_path = find_dotenv(filename=".env", usecwd=True)
            if env_path:
                load_dotenv(env_path, override=True)
        
        set_tracing_disabled(True)
        
        # Configure Azure OpenAI client
        azure_client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        set_default_openai_client(azure_client)
        set_default_openai_api("responses")
        
        self.model = os.getenv("AZURE_OPENAI_MODEL", model)
    
    def _display_tool_output(self, tool_name: str, output: any) -> None:
        """Display tool output in human-readable format."""
        console.print(f"[bold cyan]📤 {tool_name} Result:[/bold cyan]")
        
        try:
            parsed_output = output
            
            # Display in human-readable format
            if isinstance(parsed_output, dict):
                # Check for errors first
                if "error" in parsed_output:
                    console.print(f"[red]❌ Error: {parsed_output['error']}[/red]")
                    
                    if "suggestion" in parsed_output:
                        console.print(f"[blue]💡 Suggestion: {parsed_output['suggestion']}[/blue]")
                    
                    return
                
                # Handle successful responses based on tool type
                if tool_name == "search_tables":
                    self._display_search_results(parsed_output)
                elif tool_name == "get_table_info":
                    self._display_table_analysis(parsed_output)
                elif tool_name == "discover_dimension_values":
                    self._display_dimension_values(parsed_output)
                elif tool_name == "get_filtered_data":
                    self._display_filtered_data(parsed_output)
                else:
                    self._display_generic_output(parsed_output)
            else:
                console.print(f"[white]{parsed_output}[/white]")
                
        except Exception as e:
            console.print(f"[red]Error displaying output: {e}[/red]")
            console.print(f"[dim]Raw output: {output}[/dim]")
    
    def _display_search_results(self, data: dict) -> None:
        """Display search results in human-readable format"""
        query = data.get("query", "Unknown")
        tables = data.get("tables", [])
        
        console.print(f"[green]🔍 Found {len(tables)} tables for query: '{query}'[/green]")
        
        if tables:
            table = Table(title="Search Results")
            table.add_column("ID", style="cyan")
            table.add_column("Title", style="white", max_width=50)
            table.add_column("Updated", style="dim")
            table.add_column("Score", style="green")
            
            for t in tables[:5]:  # Show top 5 results
                table.add_row(
                    t.get('id', 'N/A'),
                    t.get('title', 'N/A'),
                    str(t.get('updated', 'N/A')),
                    f"{t.get('score', 'N/A')}"
                )
            
            console.print(table)
    
    def _display_table_analysis(self, data: dict) -> None:
        """Display table analysis results from get_table_info"""
        table_id = data.get("table_id", "Unknown")
        title = data.get("title", "Unknown")
        console.print(f"[green]📊 Table Analysis: {table_id}[/green]")
        console.print(f"[white]{title}[/white]")
        
        # Show time coverage
        first_period = data.get("first_period", "")
        last_period = data.get("last_period", "")
        if first_period or last_period:
            console.print(f"[dim]📅 Time Coverage: {first_period} - {last_period}[/dim]")
        
        # Show variables
        variables = data.get("variables", [])
        if variables:
            console.print(f"[cyan]📋 Dimensions ({len(variables)}):[/cyan]")
            for var in variables[:5]:  # Show first 5
                api_name = var.get("api_name", "Unknown")
                display_name = var.get("display_name", "Unknown")
                sample_count = var.get("total_values", "?")
                console.print(f"  • {api_name} ({sample_count} values)")

    def _display_dimension_values(self, data: dict) -> None:
        """Display dimension values from discover_dimension_values"""
        dimension_name = data.get("dimension_name", "Unknown")
        total_values = data.get("total_values", 0)
        shown_values = len(data.get("values", []))
        
        console.print(f"[green]🎯 Dimension Values: {dimension_name}[/green]")
        console.print(f"[dim]Showing {shown_values} of {total_values} values[/dim]")
        
        # Show sample values
        values = data.get("values", [])
        if values:
            table = Table(show_header=True, box=None)
            table.add_column("Code", style="cyan")
            table.add_column("Label", style="white", max_width=40)
            
            for value in values[:10]:  # Show first 10
                code = value.get("code", "")
                label = value.get("label", "")
                table.add_row(code, label)
            
            console.print(table)

    def _display_filtered_data(self, data: dict) -> None:
        """Display filtered data results"""
        table_id = data.get("table_id", "Unknown")
        title = data.get("title", "Unknown")
        total_points = data.get("total_data_points", 0)
        returned_points = data.get("returned_data_points", 0)
        
        console.print(f"[green]📈 Data Retrieved from {table_id}[/green]")
        console.print(f"[white]{title}[/white]")
        console.print(f"[dim]Data Points: {returned_points} of {total_points}[/dim]")
        
        # Show summary stats if available
        summary_stats = data.get("summary_stats", {})
        if summary_stats:
            console.print(f"[cyan]📊 Summary Statistics:[/cyan]")
            for key, value in summary_stats.items():
                if isinstance(value, float):
                    console.print(f"  • {key}: {value:,.2f}")
                else:
                    console.print(f"  • {key}: {value:,}")
        
        # Show sample data
        formatted_data = data.get("formatted_data", [])
        if formatted_data:
            console.print(f"[cyan]📋 Sample Data:[/cyan]")
            for i, point in enumerate(formatted_data[:5]):
                value = point.get("value")
                dimensions = {k: v for k, v in point.items() if k not in ["value", "index"]}
                dim_str = ", ".join([f"{k}={v}" for k, v in dimensions.items()])
                console.print(f"  • {value:,} ({dim_str})")

    def _display_generic_output(self, data: dict) -> None:
        """Display generic structured output in human-readable format"""
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                console.print(f"[cyan]{key}:[/cyan] [dim]{type(value).__name__} with {len(value)} items[/dim]")
            else:
                console.print(f"[cyan]{key}:[/cyan] [white]{value}[/white]")
    
    def _display_tool_result_summary(self, tool_name: str, output: any) -> None:
        """Display brief summary of tool results to show if agent is on right track."""
        try:
            # Parse the output - handle various formats
            parsed_output = self._parse_tool_output(output)
            
            if isinstance(parsed_output, dict):
                # Check for errors first
                if "error" in parsed_output:
                    console.print(f"   [red]❌ {parsed_output.get('error', 'Unknown error')}[/red]")
                    return
                
                # Tool-specific summaries
                if tool_name == "search_tables":
                    tables = parsed_output.get("tables", [])
                    total = parsed_output.get("total_found", len(tables))
                    if tables:
                        console.print(f"   [green]✓ Found {total} tables[/green]")
                        # Show top 5 tables found
                        for i, table in enumerate(tables[:5]):
                            table_id = table.get('id', 'N/A')
                            title = table.get('title', 'N/A')[:60]
                            score = table.get('score', 0)
                            console.print(f"     {i+1}. [cyan]{table_id}[/cyan] - {title}... [dim](score: {score})[/dim]")
                        if len(tables) > 5:
                            console.print(f"     [dim]... and {len(tables) - 5} more[/dim]")
                    else:
                        console.print(f"   [yellow]⚠ No tables found for search query[/yellow]")
                
                elif tool_name == "get_table_info":
                    table_id = parsed_output.get("table_id", "N/A")
                    variables = parsed_output.get("variables", [])
                    time_span = parsed_output.get("time_span", "N/A")
                    console.print(f"   [green]✓ Table {table_id}: {len(variables)} dimensions, {time_span}[/green]")
                    
                    # Show key dimensions
                    if variables:
                        dim_names = [v.get("api_name", "N/A") for v in variables[:3]]
                        console.print(f"   [dim]  Key dimensions: {', '.join(dim_names)}[/dim]")
                    
                    # Show aggregation options if available
                    agg_options = parsed_output.get("aggregation_options", {})
                    if agg_options:
                        console.print(f"   [dim]  Aggregation options available:[/dim]")
                        for dim_name, options in list(agg_options.items())[:3]:  # Show first 3
                            valuesets = options.get("valuesets", [])
                            aggregations = options.get("aggregations", [])
                            if valuesets or aggregations:
                                option_list = []
                                for vs in valuesets[:2]:  # Show first 2 of each type
                                    option_list.append(f"{vs.get('id', 'N/A')}")
                                for agg in aggregations[:2]:
                                    option_list.append(f"{agg.get('id', 'N/A')}")
                                if option_list:
                                    console.print(f"     • {dim_name}: {', '.join(option_list)}")
                
                elif tool_name == "discover_dimension_values":
                    dim_name = parsed_output.get("dimension_name", "N/A")
                    total_values = parsed_output.get("total_values", 0)
                    matching = parsed_output.get("matching_values", 0)
                    console.print(f"   [green]✓ Dimension '{dim_name}': {matching}/{total_values} values found[/green]")
                    
                    # Show sample values
                    values = parsed_output.get("values", [])
                    if values:
                        sample = [f"{v.get('code', 'N/A')}={v.get('label', 'N/A')}" for v in values[:2]]
                        console.print(f"   [dim]  Sample: {', '.join(sample)}[/dim]")
                
                elif tool_name == "get_filtered_data":
                    table_id = parsed_output.get("table_id", "N/A")
                    total_points = parsed_output.get("total_data_points", 0)
                    returned_points = parsed_output.get("returned_data_points", 0)
                    console.print(f"   [green]✓ Retrieved {returned_points}/{total_points} data points from table {table_id}[/green]")
                    
                    # Show summary stats if available
                    summary_stats = parsed_output.get("summary_stats", {})
                    if summary_stats:
                        min_val = summary_stats.get("min", 0)
                        max_val = summary_stats.get("max", 0)
                        console.print(f"   [dim]  Value range: {min_val:.2f} - {max_val:.2f}[/dim]")
                
                else:
                    # Generic summary
                    key_fields = [k for k in parsed_output.keys() if k not in ['error', 'agent_guidance']][:3]
                    console.print(f"   [green]✓ Returned data with fields: {', '.join(key_fields)}[/green]")
            
            else:
                console.print(f"   [green]✓ {tool_name} completed[/green]")
                
        except Exception as e:
            console.print(f"   [yellow]⚠ {tool_name} completed (unable to parse result)[/yellow]")
    
    def _parse_tool_output(self, output: any) -> any:
        """Parse tool output handling various formats from the agent framework."""
        try:
            # Handle MCP-style output format
            if isinstance(output, dict) and "content" in output:
                content = output["content"]
                if isinstance(content, list) and len(content) > 0:
                    first_content = content[0]
                    if isinstance(first_content, dict) and "text" in first_content:
                        try:
                            return json.loads(first_content["text"])
                        except json.JSONDecodeError:
                            return first_content["text"]
                    else:
                        return content
                else:
                    return content
            
            # Handle wrapped JSON format {"type": "text", "text": "..."}
            elif isinstance(output, dict) and output.get("type") == "text":
                text_content = output.get("text", "")
                try:
                    return json.loads(text_content)
                except json.JSONDecodeError:
                    return text_content
            
            # Handle string JSON
            elif isinstance(output, str):
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return output
            
            # Already a dict or other type
            else:
                return output
                
        except Exception:
            return output
    
    async def process_query(self, query: str) -> str:
        """Process a query using the SSB agent with autonomous tool usage and streaming."""
        t0 = time.monotonic()
        
        try:
            # Create agent with tools registered - let agent decide how to use them
            agent = Agent(
                name="SSB Statistikk-ekspert",
                instructions="""Du er en ekspert på norsk statistikk som bruker SSB API-verktøy for å besvare spørsmål.

Du har tilgang til disse verktøyene:
- search_tables: Søk etter relevante statistikktabeller
- get_table_info: Få detaljert informasjon om en tabell
- discover_dimension_values: Utforsk tilgjengelige verdier for dimensjoner
- get_filtered_data: Hent filtrerte statistikkdata

TEKNISKE KRAV:
- Bruk alltid filters_json parameter i get_filtered_data (påkrevd JSON string)
- Tidsperioder spesifiseres i filters, f.eks. '{"Tid": "2024"}'
- Bruk eksakte dimensjonsnavn fra get_table_info
- Wildcards (*) kan brukes for å få alle verdier i en dimensjon
- For aggregering kan code_lists_json brukes (f.eks. '{"NACE2007": "agg_NACE2007arb11"}')

EFFEKTIVITET:
- Minimer antall verktøykall
- Unngå valideringsfeil ved å bruke korrekte parametere
- Ett get_filtered_data kall kan ofte hente all nødvendig data

SVAR PÅ NORSK:
- Presenter resultater med norske tall og enheter
- Oppgi kilde med tabellnummer fra SSB
- Bruk norske stedsnavn og terminologi

Din oppgave er å finne relevant statistikk og presentere den på en klar og nyttig måte.""",
                model=self.model,
                tools=[search_tables, get_table_info, discover_dimension_values, get_filtered_data],
                model_settings=agent_model_settings.ModelSettings(
                    reasoning={
                        "effort": os.getenv("AZURE_REASONING_EFFORT", "medium"),
                        "summary": os.getenv("AZURE_REASONING_SUMMARY", "auto"),
                    },
                ),
            )
            
            console.print(f"[bold blue]🧠 {self.model} Analysis[/bold blue]")
            console.print()
            
            # Use streaming to show progress
            result = run.Runner.run_streamed(agent, query, max_turns=20)
            
            # Track tool usage
            tool_calls = []
            final_content = []
            first_token_time = None
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Processing...", total=None)
                
                async for event in result.stream_events():
                    if event.type == "run_item_stream_event":
                        # Handle tool calls
                        if event.item.type == "tool_call_item":
                            tool_name = 'unknown_tool'
                            tool_args = {}
                            
                            if hasattr(event.item, 'raw_item') and hasattr(event.item.raw_item, 'name'):
                                tool_name = event.item.raw_item.name
                            
                            if hasattr(event.item, 'raw_item') and hasattr(event.item.raw_item, 'arguments'):
                                try:
                                    if isinstance(event.item.raw_item.arguments, str):
                                        tool_args = json.loads(event.item.raw_item.arguments)
                                    else:
                                        tool_args = event.item.raw_item.arguments
                                except (json.JSONDecodeError, TypeError):
                                    tool_args = {"raw": str(event.item.raw_item.arguments)}
                            
                            tool_calls.append(tool_name)
                            
                            # Display tool call in a nice format
                            progress.stop()
                            console.print(f"\n[yellow]🔧 Calling: {tool_name}[/yellow]")
                            
                            # Show key parameters
                            if tool_args:
                                if tool_name == "search_tables" and "query" in tool_args:
                                    console.print(f"   [dim]→ Query: {tool_args['query']}[/dim]")
                                elif tool_name == "get_table_info" and "table_id" in tool_args:
                                    console.print(f"   [dim]→ Table: {tool_args['table_id']}[/dim]")
                                elif tool_name == "discover_dimension_values":
                                    if "dimension_name" in tool_args:
                                        console.print(f"   [dim]→ Dimension: {tool_args['dimension_name']}[/dim]")
                                elif tool_name == "get_filtered_data":
                                    if "table_id" in tool_args:
                                        console.print(f"   [dim]→ Table: {tool_args['table_id']}[/dim]")
                                    if "filters_json" in tool_args:
                                        try:
                                            filters = json.loads(tool_args['filters_json'])
                                            console.print(f"   [dim]→ Filters: {json.dumps(filters, ensure_ascii=False)}[/dim]")
                                        except:
                                            pass
                            
                            progress.start()
                            progress.update(task, description=f"Executing {tool_name}...")
                        
                        elif event.item.type == "tool_call_output_item":
                            # Tool completed - show brief summary of results
                            tool_output = event.item.output
                            last_tool = tool_calls[-1] if tool_calls else "unknown"
                            
                            progress.stop()
                            self._display_tool_result_summary(last_tool, tool_output)
                            progress.start()
                            progress.update(task, description="Processing results...")
                        
                        elif event.item.type == "message_output_item":
                            # Final message - stop progress
                            progress.stop()
                    
                    elif event.type == "raw_response_event":
                        # Handle raw response events for streaming text
                        if hasattr(event.data, 'type'):
                            if event.data.type == "response.output_text.delta":
                                if first_token_time is None:
                                    first_token_time = time.monotonic()
                                    progress.stop()
                                    console.print("\n[bold green]📋 Answer:[/bold green]")
                                
                                if hasattr(event.data, 'delta') and event.data.delta:
                                    final_content.append(event.data.delta)
                                    console.print(f"[white]{event.data.delta}[/white]", end="")
                            
                            elif event.data.type == "response.output_text.done":
                                console.print("")  # New line after streaming
            
            # Get final output
            final_output = result.final_output or ''.join(final_content)
            
            return final_output
            
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")
            return f"Error: {str(e)}"
        
        finally:
            # Show summary
            end_time = time.monotonic()
            console.print("\n" + "="*60)
            console.print("🎯 Standalone Agent Summary")
            console.print(f"⏱ Total time: {int((end_time - t0)*1000)} ms")
            console.print("="*60)

async def main():
    """Main CLI interface for standalone agent."""
    
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
    
    console.print(Panel(
        Text(query, style="bold cyan"),
        title="[bold blue]🤖 SSB Standalone Agent[/bold blue]",
        border_style="blue"
    ))
    console.print()
    
    agent = SSBStandaloneAgent()
    
    # Process query with streaming
    answer = await agent.process_query(query)

if __name__ == "__main__":
    asyncio.run(main())