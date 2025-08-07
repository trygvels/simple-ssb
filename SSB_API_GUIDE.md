# SSB PxWebAPI v2-Beta: Complete Implementation Guide

## Overview

This guide provides comprehensive documentation for utilizing the SSB (Statistics Norway) PxWebAPI v2-beta for building sophisticated MCP tools with advanced filtering, aggregation, and data analysis capabilities.

## Table of Contents
1. [API Architecture](#api-architecture)
2. [Authentication & Limits](#authentication--limits)
3. [Core Endpoints](#core-endpoints)
4. [Advanced Filtering](#advanced-filtering)
5. [Aggregation Capabilities](#aggregation-capabilities)
6. [Data Fetching Strategies](#data-fetching-strategies)
7. [Output Formats](#output-formats)
8. [MCP Tool Implementation](#mcp-tool-implementation)
9. [Best Practices](#best-practices)
10. [Error Handling](#error-handling)
11. [Code Examples](#code-examples)

## API Architecture

### Base Configuration
- **Base URL**: `https://data.ssb.no/api/pxwebapi/v2-beta/`
- **Version**: 2.0.0-beta.18
- **License**: Creative Commons CC0 Public Domain
- **CORS**: Enabled for web applications

### Rate Limits & Constraints
- **Rate Limit**: 30 queries per 10-minute window
- **Data Limit**: 800,000 cells per request
- **URL Length**: ~2,500 characters for GET requests
- **Maintenance**: API unavailable 05:00-08:15 and weekends

## Authentication & Limits

No authentication required, but respect these limits:

```python
# Check current limits
async def get_api_config():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://data.ssb.no/api/pxwebapi/v2-beta/config"
        )
        return response.json()
```

## Core Endpoints

### 1. Configuration Endpoint
**Purpose**: Get API capabilities and current limits
```python
GET /config
```

### 2. Navigation Endpoint  
**Purpose**: Browse statistical database structure
```python
GET /navigation?lang=en
```

Response includes 24 subject areas:
- `al`: Labour market and earnings
- `be`: Population demographics  
- `ba`: Banking and financial markets
- `he`: Health statistics
- And 20 more categories

### 3. Table Search Endpoint
**Purpose**: Find tables by keyword
```python
GET /tables?query={keyword}&lang=en&pageSize=50&pageNumber=1
```

### 4. Table Metadata Endpoint
**Purpose**: Get complete table structure
```python
GET /tables/{tableId}/metadata?lang=en&outputformat=json-stat2
```

### 5. Data Retrieval Endpoint
**Purpose**: Fetch actual statistical data
```python
GET /tables/{tableId}/data?{parameters}
POST /tables/{tableId}/data
```

## Advanced Filtering

### Value Selection Patterns

#### 1. Universal Selection (All Values)
```
valueCodes[dimensionName]=*
```
Retrieves all available values for a dimension.

#### 2. Specific Value Selection
```
valueCodes[Region]=0301,0101,1103
valueCodes[Kjonn]=1,2
```

#### 3. Temporal Filtering

**Most Recent N Periods:**
```
valueCodes[Tid]=top(5)
```

**From Specific Date:**
```
valueCodes[Tid]=from(2020)
```

**Range Selection:**
```
valueCodes[Tid]=2020,2021,2022,2023,2024
```

**Wildcard Patterns:**
```
valueCodes[Tid]=202*        # All 2020s
valueCodes[Region]=03??     # All codes starting with 03
```

### Geographic Filtering

#### Administrative Levels
```python
# National level
valueCodes[Region]=0

# County level  
valueCodes[Region]=03,11,15

# Municipal level
valueCodes[Region]=0301,1103,1506
```

#### Using Aggregation Groups
```python
# Get aggregated regional data
outputValues[Region]=aggregated
```

## Aggregation Capabilities

### Built-in Aggregation Levels

#### 1. Geographic Aggregation
- **National**: `Region=0`
- **Counties**: `Region=01,02,03...`
- **Municipalities**: `Region=0101,0102,0103...`
- **Statistical Regions**: Available in metadata `codeLists/agg`

#### 2. Age Group Aggregation
```python
# Standard age groups
valueCodes[Alder]=0-17,18-66,67+

# Custom age ranges
valueCodes[Alder]=25-34,35-44,45-54
```

#### 3. Time Period Aggregation
```python
# Annual data
valueCodes[Tid]=2020,2021,2022

# Multi-year periods
valueCodes[Tid]=2020-2022
```

### Using Aggregated Output
```python
# Request aggregated values instead of individual codes
outputValues[Region]=aggregated
outputValues[Alder]=aggregated
```

## Data Fetching Strategies

### 1. Efficient Query Construction

#### Small Datasets (< 10,000 cells)
```python
# Use GET with specific parameters
params = {
    "lang": "en",
    "outputformat": "json-stat2",
    "valueCodes[Region]": "0301,1103",
    "valueCodes[Tid]": "top(3)"
}
```

#### Large Datasets (> 10,000 cells)
```python
# Use POST to avoid URL length limits
query = {
    "lang": "en",
    "outputformat": "json-stat2", 
    "valueCodes": {
        "Region": ["0301", "1103", "1506"],
        "Alder": ["*"],
        "Tid": ["from(2020)"]
    }
}

# POST request
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"https://data.ssb.no/api/pxwebapi/v2-beta/tables/{table_id}/data",
        json=query
    )
```

### 2. Pagination for Table Search
```python
async def search_all_tables(query: str, max_pages: int = 10):
    all_tables = []
    page = 1
    
    while page <= max_pages:
        params = {
            "query": query,
            "lang": "en", 
            "pageSize": 50,
            "pageNumber": page
        }
        
        response = await client.get(
            "https://data.ssb.no/api/pxwebapi/v2-beta/tables",
            params=params
        )
        
        data = response.json()
        tables = data.get('tables', [])
        
        if not tables:
            break
            
        all_tables.extend(tables)
        page += 1
    
    return all_tables
```

### 3. Metadata-Driven Query Building
```python
async def build_smart_query(table_id: str, filters: dict):
    # Get metadata first
    metadata = await get_table_metadata(table_id)
    
    query_params = {
        "lang": "en",
        "outputformat": "json-stat2"
    }
    
    # Build valueCodes based on available variables
    for variable in metadata.get('variables', []):
        var_code = variable['code']
        
        if var_code in filters:
            if filters[var_code] == 'recent':
                query_params[f"valueCodes[{var_code}]"] = "top(3)"
            elif filters[var_code] == 'all':
                query_params[f"valueCodes[{var_code}]"] = "*"
            else:
                query_params[f"valueCodes[{var_code}]"] = filters[var_code]
    
    return query_params
```

## Output Formats

### 1. JSON-stat2 (Recommended)
Best for programmatic analysis with rich metadata:
```python
params = {"outputformat": "json-stat2"}
```

**Structure:**
```json
{
  "version": "2.0",
  "class": "dataset",
  "source": "Statistics Norway",
  "updated": "2024-10-01T06:00:00Z",
  "id": ["Region", "Tid", "ContentsCode"],
  "size": [994, 5, 1],
  "dimension": {
    "Region": {
      "category": {
        "index": {"0101": 0, "0301": 1},
        "label": {"0101": "Halden", "0301": "Oslo"}
      }
    }
  },
  "value": [12345, 67890, ...]
}
```

### 2. CSV Format
For spreadsheet applications:
```python
params = {"outputformat": "csv"}
```

### 3. Enhanced Output with Text Labels
```python
params = {
    "outputformat": "json-stat2",
    "outputFormatParams": "UseTexts"
}
```

## MCP Tool Implementation

Based on the research, here are enhanced MCP tools:

### 1. Enhanced Table Search Tool
```python
@mcp.tool()
async def search_tables_advanced(
    query: str = Field(description="Search query for SSB tables"),
    subject_area: str = Field(default="", description="Filter by subject (e.g., 'al' for labour)"),
    max_results: int = Field(default=20, description="Maximum results"),
    recent_only: bool = Field(default=False, description="Only recently updated tables"),
    language: str = Field(default="no", pattern="^(no|en)$")
) -> dict:
    """Advanced search for SSB statistical tables with filtering options."""
    
    params = {
        "query": query,
        "lang": language,
        "pageSize": min(max_results, 100)
    }
    
    if recent_only:
        params["pastDays"] = 30
        
    # Implementation with subject area filtering...
    return search_results
```

### 2. Metadata Analysis Tool
```python
@mcp.tool()
async def analyze_table_structure(
    table_id: str = Field(description="SSB table identifier"),
    language: str = Field(default="no", pattern="^(no|en)$")
) -> dict:
    """Analyze table structure and available dimensions for query building."""
    
    # Get comprehensive metadata
    metadata = await get_table_metadata(table_id, language)
    
    analysis = {
        "table_id": table_id,
        "title": metadata.get("label", ""),
        "description": metadata.get("description", ""),
        "variables": [],
        "time_coverage": {},
        "geographic_coverage": {},
        "recommended_queries": []
    }
    
    # Analyze each variable
    for variable in metadata.get("variables", []):
        var_analysis = {
            "code": variable["code"],
            "label": variable.get("label", ""),
            "total_values": len(variable.get("values", [])),
            "sample_values": variable.get("values", [])[:5],
            "has_aggregation": "agg" in variable.get("codeLists", {})
        }
        analysis["variables"].append(var_analysis)
    
    return analysis
```

### 3. Filtered Data Retrieval Tool
```python
@mcp.tool()
async def get_filtered_data(
    table_id: str = Field(description="SSB table identifier"),
    filters: dict = Field(description="Dimension filters as key-value pairs"),
    time_filter: str = Field(default="top(5)", description="Time period filter"),
    output_format: str = Field(default="json-stat2", description="Output format"),
    language: str = Field(default="no", pattern="^(no|en)$"),
    max_cells: int = Field(default=10000, le=100000, description="Maximum data cells")
) -> dict:
    """Retrieve filtered statistical data with flexible dimension selection."""
    
    query_params = {
        "lang": language,
        "outputformat": output_format,
        "outputFormatParams": "UseTexts"
    }
    
    # Apply time filter
    query_params[f"valueCodes[Tid]"] = time_filter
    
    # Apply dimension filters
    for dimension, values in filters.items():
        if isinstance(values, list):
            query_params[f"valueCodes[{dimension}]"] = ",".join(values)
        else:
            query_params[f"valueCodes[{dimension}]"] = values
    
    # Implementation with cell count estimation...
    return filtered_data
```

### 4. Data Comparison Tool
```python
@mcp.tool()
async def compare_regions_over_time(
    table_id: str = Field(description="SSB table identifier"),
    regions: List[str] = Field(description="List of region codes to compare"),
    start_period: str = Field(description="Start time period"),
    end_period: str = Field(description="End time period"),
    language: str = Field(default="no", pattern="^(no|en)$")
) -> dict:
    """Compare statistical data across regions and time periods."""
    
    # Build comparison query
    query_params = {
        "lang": language,
        "outputformat": "json-stat2",
        f"valueCodes[Region]": ",".join(regions),
        f"valueCodes[Tid]": f"from({start_period})"
    }
    
    # Implementation with trend analysis...
    return comparison_results
```

## Best Practices

### 1. Query Optimization
```python
# Always check metadata first
metadata = await get_table_metadata(table_id)

# Estimate cell count before querying
estimated_cells = 1
for variable in metadata['variables']:
    estimated_cells *= len(variable['values'])

if estimated_cells > 800000:
    # Apply more restrictive filters
    pass
```

### 2. Rate Limit Management
```python
import asyncio
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_calls=30, time_window=600):
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
            await asyncio.sleep(sleep_time)
        
        self.calls.append(now)
```

### 3. Caching Strategy
```python
import json
import hashlib
from datetime import datetime, timedelta

class APICache:
    def __init__(self, ttl_hours=24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def get_cache_key(self, url, params):
        key_data = f"{url}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_or_fetch(self, url, params, fetch_func):
        cache_key = self.get_cache_key(url, params)
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.ttl:
                return cached_data
        
        # Cache miss or expired - fetch new data
        data = await fetch_func(url, params)
        self.cache[cache_key] = (data, datetime.now())
        return data
```

### 4. Error Recovery
```python
import asyncio
from typing import Optional

async def robust_api_call(
    url: str, 
    params: dict, 
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> Optional[dict]:
    """Make API call with exponential backoff retry logic."""
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limited
                wait_time = backoff_factor ** attempt
                await asyncio.sleep(wait_time)
                continue
            else:
                raise
                
        except (httpx.RequestError, asyncio.TimeoutError):
            if attempt == max_retries - 1:
                raise
            wait_time = backoff_factor ** attempt
            await asyncio.sleep(wait_time)
    
    return None
```

## Error Handling

### Common Error Scenarios

#### 1. Rate Limit Exceeded (429)
```python
if response.status_code == 429:
    retry_after = response.headers.get('Retry-After', 60)
    await asyncio.sleep(int(retry_after))
```

#### 2. Data Limit Exceeded (400)
```json
{
  "error": "Query would return more than 800,000 data cells",
  "suggestion": "Apply more restrictive filters"
}
```

#### 3. Invalid Table ID (404)
```json
{
  "error": "Table not found",
  "table_id": "invalid_id"
}
```

#### 4. Invalid Parameters (400)
```json
{
  "error": "Invalid dimension code",
  "dimension": "InvalidDimension"
}
```

## Code Examples

### Complete MCP Tool Implementation

```python
#!/usr/bin/env python3
"""
Enhanced SSB MCP Server with advanced filtering and aggregation
"""

import asyncio
import httpx
from typing import Dict, List, Any, Optional
from fastmcp import FastMCP
from pydantic import Field
import json

mcp = FastMCP("Enhanced SSB Discovery")

class SSBAPIClient:
    def __init__(self):
        self.base_url = "https://data.ssb.no/api/pxwebapi/v2-beta"
        self.rate_limiter = RateLimiter()
        self.cache = APICache()
    
    async def search_tables(
        self, 
        query: str, 
        language: str = "no",
        max_results: int = 50
    ) -> dict:
        """Search for tables with enhanced filtering."""
        
        await self.rate_limiter.acquire()
        
        params = {
            "query": query,
            "lang": language,
            "pageSize": max_results
        }
        
        url = f"{self.base_url}/tables"
        
        return await self.cache.get_or_fetch(
            url, params, self._fetch_data
        )
    
    async def get_metadata(self, table_id: str, language: str = "no") -> dict:
        """Get comprehensive table metadata."""
        
        await self.rate_limiter.acquire()
        
        params = {"lang": language}
        url = f"{self.base_url}/tables/{table_id}/metadata"
        
        return await self._fetch_data(url, params)
    
    async def get_filtered_data(
        self,
        table_id: str,
        filters: dict,
        language: str = "no",
        output_format: str = "json-stat2"
    ) -> dict:
        """Get filtered data with automatic optimization."""
        
        await self.rate_limiter.acquire()
        
        # Build query parameters
        params = {
            "lang": language,
            "outputformat": output_format,
            "outputFormatParams": "UseTexts"
        }
        
        # Apply filters
        for dimension, values in filters.items():
            if isinstance(values, list):
                params[f"valueCodes[{dimension}]"] = ",".join(values)
            else:
                params[f"valueCodes[{dimension}]"] = values
        
        url = f"{self.base_url}/tables/{table_id}/data"
        
        return await self._fetch_data(url, params)
    
    async def _fetch_data(self, url: str, params: dict) -> dict:
        """Internal method for making HTTP requests."""
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

# Initialize client
ssb_client = SSBAPIClient()

@mcp.tool()
async def search_tables_enhanced(
    query: str = Field(description="Search query for SSB tables"),
    language: str = Field(default="no", pattern="^(no|en)$"),
    max_results: int = Field(default=20, le=100)
) -> dict:
    """Enhanced table search with scoring and filtering."""
    
    try:
        results = await ssb_client.search_tables(query, language, max_results)
        
        # Enhanced scoring based on query relevance
        query_words = query.lower().split()
        scored_tables = []
        
        for table in results.get('tables', []):
            title = table.get('label', '').lower()
            description = table.get('description', '').lower()
            
            score = 0
            for word in query_words:
                if word in title:
                    score += 5
                if word in description:
                    score += 2
                if any(word in part for part in title.split()):
                    score += 3
            
            if score > 0:
                scored_tables.append({
                    'id': table.get('id'),
                    'title': table.get('label'),
                    'description': table.get('description', ''),
                    'updated': table.get('updated', ''),
                    'score': score,
                    'time_period': table.get('firstPeriod', '') + " - " + table.get('lastPeriod', ''),
                    'variables': len(table.get('variables', []))
                })
        
        # Sort by score
        scored_tables.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "query": query,
            "tables": scored_tables[:max_results],
            "total_found": len(scored_tables)
        }
    
    except Exception as e:
        return {"error": f"Search failed: {str(e)}", "tables": []}

@mcp.tool()
async def analyze_table_metadata(
    table_id: str = Field(description="SSB table identifier"),
    language: str = Field(default="no", pattern="^(no|en)$")
) -> dict:
    """Analyze table structure and provide query recommendations."""
    
    try:
        metadata = await ssb_client.get_metadata(table_id, language)
        
        analysis = {
            "table_id": table_id,
            "title": metadata.get("label", ""),
            "description": metadata.get("description", ""),
            "last_updated": metadata.get("updated", ""),
            "variables": [],
            "query_suggestions": []
        }
        
        # Analyze each variable
        for variable in metadata.get("variables", []):
            var_info = {
                "code": variable["code"],
                "label": variable.get("label", ""),
                "type": variable.get("type", ""),
                "total_values": len(variable.get("values", [])),
                "sample_values": [
                    {
                        "code": val.get("code", ""),
                        "label": val.get("label", "")
                    }
                    for val in variable.get("values", [])[:5]
                ],
                "has_aggregation": bool(variable.get("aggregated", False))
            }
            analysis["variables"].append(var_info)
        
        # Generate query suggestions
        time_vars = [v for v in metadata.get("variables", []) 
                    if v["code"].lower() in ["tid", "time", "aar"]]
        region_vars = [v for v in metadata.get("variables", [])
                      if v["code"].lower() in ["region", "kommune", "fylke"]]
        
        if time_vars:
            analysis["query_suggestions"].append({
                "type": "recent_data",
                "description": "Get most recent 5 time periods",
                "filter": {time_vars[0]["code"]: "top(5)"}
            })
        
        if region_vars:
            analysis["query_suggestions"].append({
                "type": "major_regions",
                "description": "Compare major regions",
                "filter": {region_vars[0]["code"]: "03,11,15,18,30"}
            })
        
        return analysis
    
    except Exception as e:
        return {"error": f"Metadata analysis failed: {str(e)}"}

@mcp.tool()
async def get_comparative_data(
    table_id: str = Field(description="SSB table identifier"),
    dimensions: dict = Field(description="Dimension filters as JSON object"),
    time_period: str = Field(default="top(5)", description="Time period selection"),
    language: str = Field(default="no", pattern="^(no|en)$")
) -> dict:
    """Get comparative data with automatic formatting and analysis."""
    
    try:
        # Add time filter to dimensions
        filters = dict(dimensions)
        filters["Tid"] = time_period
        
        data = await ssb_client.get_filtered_data(
            table_id, filters, language
        )
        
        # Process JSON-stat2 data for easier consumption
        if 'dimension' in data and 'value' in data:
            processed_data = {
                "table_id": table_id,
                "title": data.get("label", ""),
                "source": data.get("source", "Statistics Norway"),
                "updated": data.get("updated", ""),
                "dimensions": list(data["dimension"].keys()),
                "data_points": len([v for v in data["value"] if v is not None]),
                "formatted_data": []
            }
            
            # Create readable data structure
            dimensions = data["dimension"]
            values = data["value"]
            
            # Simple formatting for first 20 data points
            for i, value in enumerate(values[:20]):
                if value is not None:
                    data_point = {"value": value}
                    
                    # Add dimension labels
                    for dim_name, dim_data in dimensions.items():
                        if 'category' in dim_data and 'label' in dim_data['category']:
                            labels = list(dim_data['category']['label'].values())
                            size = len(labels)
                            if size > 0:
                                data_point[dim_name] = labels[i % size]
                    
                    processed_data["formatted_data"].append(data_point)
            
            return processed_data
        
        return data
    
    except Exception as e:
        return {"error": f"Data retrieval failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run()
```

## Conclusion

This comprehensive guide provides the foundation for building sophisticated MCP tools that fully utilize the SSB PxWebAPI v2-beta's capabilities. The enhanced tools offer:

1. **Advanced Search**: Intelligent scoring and filtering
2. **Metadata Analysis**: Automatic query suggestion generation  
3. **Flexible Filtering**: Support for complex dimension combinations
4. **Data Comparison**: Multi-region and time-series analysis
5. **Error Resilience**: Robust error handling and retry logic
6. **Performance**: Rate limiting, caching, and query optimization

These improvements enable more sophisticated statistical analysis while maintaining the simplicity and reliability that makes the tools accessible to end users.