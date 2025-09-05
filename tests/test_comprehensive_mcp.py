#!/usr/bin/env python3
"""
Comprehensive MCP Test - Single log file with all tool outputs
Tests all 5 streamlined tools and logs outputs for agent evaluation
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_server import (
    mcp,
    search_tables,
    get_table_info,
    discover_dimension_values,
    get_filtered_data
)

# Setup single log file
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"mcp_tool_outputs_{TIMESTAMP}.log"

def log_tool_output(tool_name: str, inputs: dict, output: dict, success: bool = True):
    """Log tool input/output in old format for analysis"""
    status = "SUCCESS" if success else "ERROR"
    analysis = analyze_output(output)
    
    # Write to single log file in old format
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        
        f.write(f"{timestamp_str} | INFO     | {tool_name:<25} | TOOL_CALL | Status: {status}\n")
        f.write(f"{timestamp_str} | INFO     | {tool_name:<25} | INPUTS | {json.dumps(inputs, indent=2, ensure_ascii=False)}\n")
        f.write(f"{timestamp_str} | INFO     | {tool_name:<25} | OUTPUT | {json.dumps(output, indent=2, ensure_ascii=False)}\n")
        f.write(f"{timestamp_str} | INFO     | {tool_name:<25} | ANALYSIS | {json.dumps(analysis, indent=2, ensure_ascii=False)}\n")
        f.write(f"{timestamp_str} | INFO     | {tool_name:<25} | {'='*80}\n")
    
    print(f"📝 Logged {tool_name}: {analysis['utility_score']}/10 (Agent Usability: {analysis['agent_usability']})")

def analyze_output(output: dict) -> dict:
    """Analyze tool output for agent usability using old format"""
    
    # Basic analysis like the old format
    top_level_keys = list(output.keys()) if isinstance(output, dict) else []
    has_error = "error" in output
    has_data = any(key in output for key in ["tables", "values", "formatted_data", "dimensions", "variables", "total_values", "total_found"])
    has_guidance = any(key in output for key in ["agent_guidance", "workflow_guidance", "search_tips", "recommendations", "workflow_hint", "usage_suggestion"])
    
    # Determine agent usability category
    if has_error and "suggestion" in output:
        agent_usability = "helpful_error_with_guidance"
        utility_score = 7
    elif has_error:
        agent_usability = "unhelpful_error"
        utility_score = 3
    elif has_data and has_guidance:
        agent_usability = "excellent_data_with_guidance"
        utility_score = 10
    elif has_data:
        agent_usability = "good_data_response"
        utility_score = 8
    elif has_guidance:
        agent_usability = "guidance_only"
        utility_score = 6
    else:
        agent_usability = "minimal_response"
        utility_score = 4
    
    # Bonus scoring for agent-friendly features
    if "api_name" in str(output):  # Enhanced variable info
        utility_score = min(10, utility_score + 1)
    if "next_suggested_tools" in str(output):  # Workflow guidance
        utility_score = min(10, utility_score + 1)
    if "pattern_hint" in str(output):  # Learning hints
        utility_score = min(10, utility_score + 1)
    
    return {
        "output_type": type(output).__name__,
        "top_level_keys": top_level_keys,
        "has_error": has_error,
        "has_data": has_data,
        "has_guidance": has_guidance,
        "agent_usability": agent_usability,
        "utility_score": utility_score,
        "agent_ready": utility_score >= 7  # Whether output is sufficient for agent use
    }

async def test_all_tools():
    """Test all MCP tools and log outputs for evaluation"""
    
    print("\n🚀 COMPREHENSIVE MCP TOOL TESTING & LOGGING")
    print("=" * 60)
    print(f"📝 Logging all outputs to: {LOG_FILE}")
    
    # Test scenarios
    test_queries = ["befolkning", "sysselsatte", "bolig"]
    
    for i, query in enumerate(test_queries):
        print(f"\n📊 Test scenario {i+1}: '{query}'")
        print("-" * 40)
        
        # 1. Search for tables
        print("1️⃣ Testing search_tables...")
        search_inputs = {"query": query}
        search_result = await search_tables.fn(**search_inputs)
        log_tool_output("search_tables", search_inputs, search_result, "error" not in search_result)
        
        # Get table ID for other tests
        table_id = None
        if search_result.get("tables") and len(search_result["tables"]) > 0:
            table_id = search_result["tables"][0]["id"]
            print(f"   ✅ Found tables, using {table_id}")
        else:
            print("   ❌ No tables found, skipping dependent tests")
            continue
            
        # 2. Get table info
        print("2️⃣ Testing get_table_info...")
        table_info_inputs = {"table_id": table_id, "include_structure": True}
        table_info_result = await get_table_info.fn(**table_info_inputs)
        log_tool_output("get_table_info", table_info_inputs, table_info_result, "error" not in table_info_result)
        
        if "error" in table_info_result:
            print("   ❌ Table info failed, skipping dependent tests")
            continue
            
        print(f"   ✅ Got info for '{table_info_result.get('title', 'Unknown')[:50]}...'")
        
        # Get dimension for testing
        dimension_name = None
        variables = table_info_result.get("variables", [])
        if variables:
            dimension_name = variables[0].get("api_name") or variables[0].get("display_name")
            print(f"   🎯 Testing dimension: {dimension_name}")
        
        if not dimension_name:
            print("   ❌ No dimensions found, skipping dependent tests")
            continue
        
        # 3. Discover dimension values
        print("3️⃣ Testing discover_dimension_values...")
        discover_inputs = {
            "table_id": table_id,
            "dimension_name": dimension_name,
            "search_term": "",
            "include_code_lists": True
        }
        discover_result = await discover_dimension_values.fn(**discover_inputs)
        log_tool_output("discover_dimension_values", discover_inputs, discover_result, "error" not in discover_result)
        
        if "error" not in discover_result:
            values_found = discover_result.get("total_values", 0)
            print(f"   ✅ Found {values_found} values for {dimension_name}")
            
            # Get sample code for filtering test
            sample_code = None
            values = discover_result.get("values", [])
            if values and len(values) > 0:
                sample_code = values[0].get("code")
                print(f"   🔍 Sample code: {sample_code}")
        else:
            print(f"   ❌ Discover failed: {discover_result.get('error')}")
            continue
        
        # 4. Get filtered data (if we have a sample code)
        if sample_code:
            print("4️⃣ Testing get_filtered_data...")
            filter_inputs = {
                "table_id": table_id,
                "filters": {dimension_name: sample_code}
            }
            filter_result = await get_filtered_data.fn(**filter_inputs)
            log_tool_output("get_filtered_data", filter_inputs, filter_result, "error" not in filter_result)
            
            if "error" not in filter_result:
                data_points = filter_result.get("total_data_points", 0)
                print(f"   ✅ Retrieved {data_points} data points")
            else:
                print(f"   ❌ Filtered data failed: {filter_result.get('error')}")
        
        # Brief pause between scenarios
        if i < len(test_queries) - 1:
            await asyncio.sleep(0.5)
    
    
    # Error handling tests
    print("\n⚠️ Testing error handling...")
    
    # Test with non-existent table
    print("❌ Testing non-existent table...")
    error_inputs = {"table_id": "NONEXISTENT"}
    error_result = await get_table_info.fn(**error_inputs)
    log_tool_output("get_table_info", error_inputs, error_result, False)
    
    # Test with wrong dimension name (use first successful table_id)
    if table_id:
        print("❌ Testing wrong dimension name...")
        wrong_dim_inputs = {
            "table_id": table_id,
            "dimension_name": "WRONG_DIMENSION"
        }
        wrong_dim_result = await discover_dimension_values.fn(**wrong_dim_inputs)
        log_tool_output("discover_dimension_values", wrong_dim_inputs, wrong_dim_result, False)
    
    print(f"\n✅ COMPREHENSIVE TESTING COMPLETE")
    print(f"📁 Single log file: {LOG_FILE}")
    print(f"🎯 Tested all 4 streamlined tools across multiple scenarios")
    
    print(f"\n📊 TOOL OUTPUT EVALUATION:")
    print(f"📋 Review {LOG_FILE} to evaluate if outputs provide:")
    print(f"   ✅ Clear table content and structure information")
    print(f"   ✅ Actionable filtering and querying guidance")
    print(f"   ✅ Helpful error handling with learning hints")
    print(f"   ✅ Agent workflow guidance and next steps")

if __name__ == "__main__":
    asyncio.run(test_all_tools())