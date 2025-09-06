# SSB Agent Code Interpreter Implementation Plan

## Overview

This document outlines a comprehensive plan to integrate code interpreter capabilities into the SSB standalone agent, enabling advanced data analysis, calculations, and visualizations of Norwegian statistical data.

## Research Summary

### Current State Analysis

**OpenAI Agents SDK Code Interpreter Status (2025):**
- ❌ No native code interpreter support in OpenAI Agents SDK
- ❌ GitHub issue #360 marked as "not planned" 
- ❌ No built-in sandboxed Python execution
- ✅ Function tools support with automatic schema generation

**Available Solutions:**
1. **Pyodide + WebAssembly** (Recommended)
2. **MCP Run Python Server** (Emerging)
3. **Docker-based Sandboxing**
4. **RestrictedPython** (Limited)

### SSB Data Characteristics

**Current Data Format:**
```python
{
    "table_id": "04232",
    "dimensions": ["Region", "ContentsCode", "Tid"],
    "formatted_data": [
        {"value": 123456, "Region": "Oslo", "ContentsCode": "Population", "Tid": "2024"},
        {"value": 87654, "Region": "Bergen", "ContentsCode": "Population", "Tid": "2024"}
    ],
    "summary_stats": {
        "count": 100,
        "min": 1000,
        "max": 700000,
        "average": 50000
    }
}
```

**Typical Analysis Needs:**
- ✅ Time series calculations (growth rates, trends)
- ✅ Regional comparisons and rankings
- ✅ Cross-tabulation and aggregations
- ✅ Statistical calculations (correlations, percentiles)
- ✅ Data visualization (charts, graphs)
- ✅ Multi-table joins and merges

## Implementation Architecture

### Option 1: Pyodide + LangChain Sandbox (Recommended)

**Architecture:**
```
SSB Standalone Agent
├── Standard MCP Tools (search, analyze, get_data)
├── Code Interpreter Tool (Pyodide-based)
└── Data Processing Pipeline
    ├── SSB Data → Pandas DataFrame
    ├── Code Execution (sandboxed)
    └── Results → Formatted Response
```

**Key Components:**

1. **Sandboxed Execution Environment**
   - Pyodide WebAssembly Python runtime
   - Pre-installed libraries: pandas, numpy, matplotlib, seaborn
   - Isolated from host filesystem and network
   - Stateful sessions for multi-step analysis

2. **SSB Data Adapter**
   - Convert SSB JSON-stat format to pandas DataFrames
   - Handle dimension labels and metadata
   - Support multi-table operations

3. **Code Interpreter Tool**
   - Function tool integrated with OpenAI Agents SDK
   - Natural language → Python code generation
   - Error handling and retry logic
   - Output formatting and visualization

### Option 2: MCP Run Python Server Integration

**Architecture:**
```
SSB Standalone Agent
├── MCP Server (SSB Tools)
├── MCP Server (Python Execution)
└── Agent Orchestration
```

**Benefits:**
- Leverages emerging Pydantic MCP ecosystem
- Automatic dependency management
- Standard MCP protocol compliance

**Challenges:**
- Newer technology, less mature
- Requires Deno runtime dependency
- Limited community examples

### Option 3: Docker-based Sandboxing

**Architecture:**
```
SSB Standalone Agent
├── Docker Container (Python + Libraries)
├── Volume Mounts (Data Exchange)
└── Process Management
```

**Benefits:**
- Strong isolation guarantees
- Full Python ecosystem access
- Proven security model

**Challenges:**
- Higher resource overhead
- Complex deployment requirements
- Slower execution times

## Recommended Implementation: Pyodide + LangChain Sandbox

### Phase 1: Core Integration

#### 1.1 Dependencies and Setup

```python
# New dependencies to add to requirements.txt
langchain-sandbox>=0.1.0
pyodide-py>=0.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.17.0
```

#### 1.2 Code Interpreter Tool Implementation

```python
from langchain_sandbox import PyodideSandbox
from agents import function_tool
import pandas as pd
import json
import base64
from typing import Dict, Any, Optional

@function_tool
async def execute_python_analysis(
    code: str,
    data_context: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute Python code for data analysis with SSB data context.
    
    Args:
        code: Python code to execute
        data_context: JSON string containing SSB data from previous tool calls
        session_id: Optional session ID for stateful execution
    
    Returns:
        Dict containing execution results, outputs, and any visualizations
    """
    
    sandbox = PyodideSandbox(
        allow_net=False,  # No network access for security
        stateful=True if session_id else False
    )
    
    # Prepare execution context
    setup_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Helper functions for SSB data
def ssb_to_dataframe(ssb_data):
    \"\"\"Convert SSB formatted data to pandas DataFrame\"\"\"
    if isinstance(ssb_data, str):
        ssb_data = json.loads(ssb_data)
    
    if 'formatted_data' in ssb_data:
        df = pd.DataFrame(ssb_data['formatted_data'])
        return df
    return pd.DataFrame()

def calculate_growth_rate(df, value_col='value', time_col='Tid'):
    \"\"\"Calculate period-over-period growth rates\"\"\"
    df_sorted = df.sort_values(time_col)
    df_sorted['growth_rate'] = df_sorted[value_col].pct_change() * 100
    return df_sorted

def rank_regions(df, value_col='value', region_col='Region'):
    \"\"\"Rank regions by value\"\"\"
    return df.groupby(region_col)[value_col].sum().sort_values(ascending=False)

# Load SSB data if provided
ssb_data = None
df = pd.DataFrame()
"""
    
    if data_context:
        setup_code += f"\nssb_data = {data_context}\ndf = ssb_to_dataframe(ssb_data)\n"
    
    # Execute setup + user code
    full_code = setup_code + "\n" + code
    
    try:
        result = await sandbox.execute(full_code)
        
        # Extract outputs and visualizations
        execution_result = {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "session_id": session_id
        }
        
        # Handle matplotlib plots
        if "plt.show()" in code or "plt.savefig" in code:
            execution_result["visualization_type"] = "matplotlib"
            # Note: Actual image extraction would require additional setup
        
        # Handle data outputs
        if "df" in result.namespace:
            output_df = result.namespace["df"]
            if hasattr(output_df, 'to_json'):
                execution_result["dataframe_output"] = output_df.to_json(orient='records')
        
        return execution_result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": str(e),
            "session_id": session_id
        }
```

#### 1.3 Agent Integration

```python
class SSBAnalysisAgent:
    def __init__(self):
        # ... existing initialization ...
        
        # Add code interpreter tool
        self.tools = [
            search_tables_advanced,
            analyze_table_structure,
            get_filtered_data,
            execute_python_analysis  # New tool
        ]
    
    async def process_query(self, query: str) -> str:
        # Enhanced instructions for code interpreter
        enhanced_instructions = """
        You are an expert Norwegian statistician with Python data analysis capabilities.
        
        Available tools:
        1. search_tables_advanced - Find relevant SSB tables
        2. analyze_table_structure - Understand table dimensions
        3. get_filtered_data - Retrieve statistical data
        4. execute_python_analysis - Perform calculations and analysis
        
        ANALYSIS WORKFLOW:
        1. Use SSB tools to gather data
        2. Use execute_python_analysis for:
           - Complex calculations (growth rates, percentages)
           - Statistical analysis (correlations, regressions)
           - Data transformations and aggregations
           - Rankings and comparisons
           - Time series analysis
           - Visualizations when requested
        
        DATA PROCESSING PATTERNS:
        - Pass SSB data as data_context parameter
        - Use helper functions: ssb_to_dataframe(), calculate_growth_rate(), rank_regions()
        - Always validate results and provide context
        
        PYTHON CODE EXAMPLES:
        ```python
        # Calculate population growth rates
        growth_df = calculate_growth_rate(df, 'value', 'Tid')
        print(f"Average growth rate: {growth_df['growth_rate'].mean():.2f}%")
        
        # Rank regions by population
        rankings = rank_regions(df, 'value', 'Region')
        print("Top 5 regions:", rankings.head())
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        df.groupby('Tid')['value'].sum().plot()
        plt.title('Population Over Time')
        plt.show()
        ```
        """
        
        agent = Agent(
            name="SSB Analysis Expert with Code Interpreter",
            instructions=enhanced_instructions,
            model=self.model,
            tools=self.tools
        )
        
        # ... rest of processing logic ...
```

### Phase 2: Advanced Features

#### 2.1 Multi-table Analysis

```python
@function_tool
async def merge_ssb_tables(
    table_data_list: List[str],
    join_keys: List[str],
    analysis_code: str
) -> Dict[str, Any]:
    """
    Merge multiple SSB tables and perform analysis.
    
    Args:
        table_data_list: List of JSON strings containing SSB data
        join_keys: List of column names to join on
        analysis_code: Python code for analysis
    """
    # Implementation for multi-table operations
    pass
```

#### 2.2 Visualization Enhancements

```python
@function_tool
async def create_ssb_visualization(
    data_context: str,
    chart_type: str,
    title: str,
    x_axis: str,
    y_axis: str
) -> Dict[str, Any]:
    """
    Create standardized visualizations for SSB data.
    
    Supported chart types:
    - line_chart: Time series data
    - bar_chart: Categorical comparisons
    - heatmap: Regional data
    - scatter_plot: Correlations
    """
    # Implementation for standardized charts
    pass
```

#### 2.3 Statistical Analysis Tools

```python
@function_tool
async def statistical_analysis(
    data_context: str,
    analysis_type: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform statistical analysis on SSB data.
    
    Analysis types:
    - correlation: Correlation analysis between variables
    - regression: Linear/multiple regression
    - trend_analysis: Time series trend detection
    - seasonality: Seasonal decomposition
    - forecasting: Simple forecasting models
    """
    # Implementation for statistical analyses
    pass
```

### Phase 3: Production Features

#### 3.1 Error Handling and Recovery

```python
class CodeInterpreterError(Exception):
    """Custom exception for code interpreter errors"""
    pass

async def safe_code_execution(code: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute code with comprehensive error handling.
    """
    try:
        # Validate code for dangerous operations
        forbidden_patterns = [
            'import os', 'import subprocess', 'import sys',
            'exec(', 'eval(', 'open(', '__import__'
        ]
        
        for pattern in forbidden_patterns:
            if pattern in code:
                raise CodeInterpreterError(f"Forbidden operation detected: {pattern}")
        
        # Execute with timeout
        return await asyncio.wait_for(
            execute_python_analysis(code, json.dumps(context)),
            timeout=30.0
        )
        
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": "Code execution timed out (30s limit)",
            "suggestion": "Simplify the analysis or break into smaller steps"
        }
    except CodeInterpreterError as e:
        return {
            "success": False,
            "error": f"Security violation: {str(e)}",
            "suggestion": "Use only data analysis operations"
        }
```

#### 3.2 Session Management

```python
class CodeInterpreterSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.sandbox = None
        self.data_cache = {}
    
    async def initialize(self):
        self.sandbox = PyodideSandbox(
            allow_net=False,
            stateful=True
        )
    
    async def execute(self, code: str, data_context: Optional[Dict] = None):
        # Stateful execution with data persistence
        pass
    
    async def cleanup(self):
        if self.sandbox:
            await self.sandbox.close()
```

### Phase 4: Testing and Validation

#### 4.1 Test Scenarios

```python
# Test cases for code interpreter integration
test_scenarios = [
    {
        "name": "population_growth_calculation",
        "query": "Beregn befolkningsvekst i Oslo de siste 10 årene",
        "expected_tools": ["search_tables_advanced", "get_filtered_data", "execute_python_analysis"],
        "validation": "Should calculate year-over-year growth rates"
    },
    {
        "name": "regional_comparison",
        "query": "Sammenlign sysselsetting mellom norske fylker",
        "expected_tools": ["search_tables_advanced", "get_filtered_data", "execute_python_analysis"],
        "validation": "Should rank regions and calculate statistics"
    },
    {
        "name": "correlation_analysis",
        "query": "Er det sammenheng mellom utdanning og inntekt?",
        "expected_tools": ["search_tables_advanced", "get_filtered_data", "merge_ssb_tables", "statistical_analysis"],
        "validation": "Should perform correlation analysis"
    }
]
```

#### 4.2 Performance Benchmarks

```python
async def benchmark_code_interpreter():
    """
    Benchmark code interpreter performance:
    - Startup time (sandbox initialization)
    - Execution time for common operations
    - Memory usage patterns
    - Error recovery time
    """
    benchmarks = {
        "sandbox_startup": 0,
        "dataframe_creation": 0,
        "simple_calculation": 0,
        "visualization_generation": 0,
        "statistical_analysis": 0
    }
    
    # Run benchmarks and collect metrics
    return benchmarks
```

## Implementation Timeline

### Week 1: Foundation
- [ ] Set up Pyodide + LangChain Sandbox integration
- [ ] Implement basic `execute_python_analysis` function tool
- [ ] Test SSB data → DataFrame conversion
- [ ] Basic error handling and security measures

### Week 2: Core Features  
- [ ] Integrate code interpreter with SSB standalone agent
- [ ] Implement helper functions for SSB data analysis
- [ ] Add session management for stateful execution
- [ ] Create comprehensive test suite

### Week 3: Advanced Analysis
- [ ] Multi-table merge capabilities
- [ ] Statistical analysis tools
- [ ] Visualization generation
- [ ] Performance optimization

### Week 4: Production Ready
- [ ] Security hardening and validation
- [ ] Error recovery mechanisms
- [ ] Documentation and examples
- [ ] Performance benchmarking

## Security Considerations

### Sandboxing Requirements
1. **Process Isolation**: Pyodide runs in WebAssembly, isolated from host
2. **Network Restrictions**: No network access during code execution
3. **Filesystem Protection**: No access to host filesystem
4. **Resource Limits**: Memory and CPU time limits
5. **Code Validation**: Block dangerous operations before execution

### Security Checklist
- [ ] Validate all user input and generated code
- [ ] Implement execution timeouts (30 seconds max)
- [ ] Block dangerous imports and functions
- [ ] Monitor resource usage
- [ ] Log all code executions for audit
- [ ] Implement rate limiting for code execution requests

## Performance Considerations

### Optimization Strategies
1. **Lazy Loading**: Initialize sandbox only when needed
2. **Code Caching**: Cache common analysis patterns
3. **Data Streaming**: Stream large datasets instead of loading entirely
4. **Parallel Execution**: Multiple sandboxes for concurrent requests
5. **Memory Management**: Clean up sessions after use

### Expected Performance
- **Sandbox Startup**: 2-5 seconds (WebAssembly compilation)
- **Simple Analysis**: 100-500ms (calculations on <10K rows)
- **Complex Analysis**: 1-10 seconds (statistical operations)
- **Visualizations**: 2-5 seconds (chart generation)
- **Memory Usage**: 50-200MB per session

## Examples and Use Cases

### Use Case 1: Population Growth Analysis
```python
# User Query: "Hvor mye har befolkningen i Oslo vokst siden 2020?"

# Agent workflow:
# 1. search_tables_advanced("befolkning Oslo")
# 2. get_filtered_data(table_id="07459", filters={"Region": "Oslo", "Tid": "*"})
# 3. execute_python_analysis(code="""
# # Calculate population growth since 2020
# oslo_data = df[df['Region'] == 'Oslo'].copy()
# oslo_data['Tid'] = pd.to_datetime(oslo_data['Tid'], format='%Y')
# oslo_data = oslo_data[oslo_data['Tid'] >= '2020-01-01']
# 
# # Calculate year-over-year growth
# oslo_data = oslo_data.sort_values('Tid')
# oslo_data['growth_rate'] = oslo_data['value'].pct_change() * 100
# 
# total_growth = ((oslo_data['value'].iloc[-1] / oslo_data['value'].iloc[0]) - 1) * 100
# avg_annual_growth = oslo_data['growth_rate'].mean()
# 
# print(f"Total population growth since 2020: {total_growth:.2f}%")
# print(f"Average annual growth rate: {avg_annual_growth:.2f}%")
# """, data_context=json.dumps(ssb_data))
```

### Use Case 2: Regional Employment Comparison
```python
# User Query: "Hvilke fylker har høyest sysselsetting i 2024?"

# Agent workflow:
# 1. search_tables_advanced("sysselsetting fylker")
# 2. get_filtered_data(table_id="09817", filters={"Region": "*", "Tid": "2024"})  
# 3. execute_python_analysis(code="""
# # Rank counties by employment
# employment_2024 = df[df['Tid'] == '2024'].copy()
# 
# # Group by region and sum employment
# regional_employment = employment_2024.groupby('Region')['value'].sum().sort_values(ascending=False)
# 
# # Calculate percentages
# total_employment = regional_employment.sum()
# employment_pct = (regional_employment / total_employment * 100).round(2)
# 
# # Display top 10
# top_regions = regional_employment.head(10)
# print("Top 10 fylker med høyest sysselsetting 2024:")
# for i, (region, employment) in enumerate(top_regions.items(), 1):
#     pct = employment_pct[region]
#     print(f"{i}. {region}: {employment:,.0f} ({pct}% av total)")
# """, data_context=json.dumps(ssb_data))
```

### Use Case 3: Time Series Visualization  
```python
# User Query: "Vis utvikling i arbeidsledighet over tid"

# Agent workflow includes visualization:
# execute_python_analysis(code="""
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # Prepare unemployment data
# unemployment_df = df.copy()
# unemployment_df['Tid'] = pd.to_datetime(unemployment_df['Tid'], format='%Y')
# 
# # Create time series plot
# plt.figure(figsize=(12, 6))
# 
# # Plot unemployment rate over time
# for region in unemployment_df['Region'].unique()[:5]:  # Top 5 regions
#     region_data = unemployment_df[unemployment_df['Region'] == region]
#     plt.plot(region_data['Tid'], region_data['value'], 
#              marker='o', label=region, linewidth=2)
# 
# plt.title('Arbeidsledighet over tid - Utvalgte regioner', fontsize=14, fontweight='bold')
# plt.xlabel('År', fontsize=12)
# plt.ylabel('Arbeidsledighetsrate (%)', fontsize=12)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# 
# # Display statistics
# overall_trend = unemployment_df.groupby('Tid')['value'].mean()
# print(f"Gjennomsnittlig arbeidsledighet (siste år): {overall_trend.iloc[-1]:.2f}%")
# print(f"Endring fra året før: {(overall_trend.iloc[-1] - overall_trend.iloc[-2]):.2f} prosentpoeng")
# 
# plt.show()
# """, data_context=json.dumps(ssb_data))
```

## Alternative Implementation Options

### Option A: Jupyter Kernel Integration
- Use `jupyter-client` to spawn Python kernels
- Full Python ecosystem access
- Mature notebook-style execution model
- **Challenges**: Security, resource management, complexity

### Option B: Custom Python Sandbox
- Build custom secure Python environment
- Fine-grained control over execution
- Optimized for SSB data operations  
- **Challenges**: Security vulnerabilities, maintenance overhead

### Option C: WebAssembly + Micropython
- Ultra-lightweight Python subset
- Very fast startup times
- Limited library ecosystem
- **Challenges**: Library compatibility, feature limitations

## Conclusion

The **Pyodide + LangChain Sandbox** approach represents the optimal balance of:
- ✅ **Security**: Strong WebAssembly isolation
- ✅ **Performance**: Reasonable execution times  
- ✅ **Compatibility**: Full pandas/numpy/matplotlib support
- ✅ **Maintainability**: Proven open-source solution
- ✅ **Integration**: Clean function tool implementation

This implementation will transform the SSB agent from a data retrieval tool into a comprehensive statistical analysis platform, enabling users to perform complex calculations, generate insights, and create visualizations directly through natural language queries.

The phased approach ensures steady progress while maintaining the existing system's reliability and performance. Security remains paramount throughout the implementation, with multiple layers of protection against code injection and system compromise.

---

**Implementation Priority**: HIGH  
**Estimated Effort**: 3-4 weeks full-time development  
**Risk Level**: MEDIUM (well-established technologies, clear requirements)  
**Impact**: VERY HIGH (revolutionary enhancement to agent capabilities)