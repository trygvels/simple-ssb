#!/usr/bin/env python3
"""
Test FoU (Research and Development) domain - new domain not previously tested
Target: Table 13517 for EU-financing of institute sector R&D
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ssb_agent_mcp import SSBAgent
from rich.console import Console

console = Console()

async def test_fou_domain():
    """Test Research & Development domain - completely new statistical area"""
    
    console.print("\n🔬 Testing FoU (Research & Development) Domain")
    console.print("=" * 70)
    
    # Complex FoU query targeting table 13517
    test_query = "hvor mye av fou utgifter i instituttsektoren ble finansiert av eu-sektoren for Primærnæringsinstitutter?"
    
    console.print(f"🎯 Target: Table 13517 (FoU-utgifter instituttsektoren)")
    console.print(f"🔍 Query: [cyan]{test_query}[/cyan]")
    console.print("📊 Testing: New domain (R&D), complex financing question")
    console.print("-" * 70)
    
    # Track performance
    tool_call_count = 0
    validation_errors = 0
    found_target_table = False
    
    class FoUTracker:
        def __init__(self, original_display):
            self.original_display = original_display
            
        def __call__(self, tool_name, output):
            nonlocal tool_call_count, validation_errors, found_target_table
            tool_call_count += 1
            
            # Check for validation errors
            if "validation error" in str(output).lower() or "required property" in str(output).lower():
                validation_errors += 1
                console.print(f"❌ Validation error in {tool_name}")
            
            # Check if we found the target table
            if "13517" in str(output):
                found_target_table = True
                console.print(f"🎯 Found target table 13517 in {tool_name}!")
            
            console.print(f"📤 Tool call #{tool_call_count}: {tool_name}")
            
            # Show key information for analysis
            if tool_name == "search_tables" and isinstance(output, dict):
                tables = output.get("tables", [])
                console.print(f"   📋 Found {len(tables)} tables")
                # Check if 13517 is in results
                for table in tables[:3]:
                    table_id = table.get("id", "")
                    title = table.get("title", "")[:80]
                    if table_id == "13517":
                        console.print(f"   🎯 Target found: {table_id} - {title}")
                    else:
                        console.print(f"   • {table_id} - {title}")
            
            elif tool_name == "get_table_info" and isinstance(output, dict):
                table_id = output.get("table_id", "")
                title = output.get("title", "")
                console.print(f"   📊 Analyzing: {table_id} - {title[:60]}...")
            
            elif tool_name == "get_filtered_data" and isinstance(output, dict):
                if "error" not in output:
                    data_points = output.get("total_data_points", 0)
                    console.print(f"   📈 Retrieved {data_points} data points")
                else:
                    console.print(f"   ❌ Error: {output.get('error', 'Unknown')}")
            
            return self.original_display(tool_name, output)
    
    # Run the test
    agent = SSBAgent()
    agent._display_tool_output = FoUTracker(agent._display_tool_output)
    
    start_time = time.time()
    
    try:
        result = await asyncio.wait_for(agent.process_query(test_query), timeout=60.0)
        success = True
        end_time = time.time()
        
        console.print(f"\n✅ FoU Domain Test Results:")
        console.print(f"   • Total tool calls: {tool_call_count}")
        console.print(f"   • Validation errors: {validation_errors}")
        console.print(f"   • Found target table 13517: {found_target_table}")
        console.print(f"   • Duration: {end_time - start_time:.1f}s")
        console.print(f"   • Success: {success}")
        
        # Evaluate domain coverage
        if found_target_table and tool_call_count <= 5 and validation_errors == 0:
            console.print("\n🎉 [green]DOMAIN COVERAGE EXCELLENT![/green]")
            console.print("   Agent successfully navigated new FoU domain!")
        elif found_target_table:
            console.print("\n⚡ [yellow]DOMAIN COVERAGE GOOD[/yellow]")
            console.print(f"   Found target but needed {tool_call_count} calls")
        else:
            console.print("\n❓ [orange]DOMAIN EXPLORATION[/orange]")
            console.print("   Agent explored domain but may have found alternative table")
        
        # Show result quality
        if result and len(result) > 100:
            console.print(f"\n📝 Final answer:")
            console.print(f"   Length: {len(result)} characters")
            # Show first part of answer
            preview = result[:200] + "..." if len(result) > 200 else result
            console.print(f"   Preview: {preview}")
    
    except asyncio.TimeoutError:
        console.print("⏰ Test timed out")
    except Exception as e:
        console.print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_fou_domain())