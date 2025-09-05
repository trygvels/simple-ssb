#!/usr/bin/env python3
"""
Test script to validate the optimized agent behavior
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ssb_agent_mcp import SSBAgent
from rich.console import Console

console = Console()

async def test_optimization():
    """Test the optimized agent with the problematic query from user example"""
    
    console.print("\n🧪 Testing Optimized Agent Behavior")
    console.print("=" * 60)
    
    # Test the exact same query that was inefficient before
    test_query = "Hvilken næring har flest sysselsatte? Gi meg top 5 i 2024"
    
    console.print(f"🔍 Query: [cyan]{test_query}[/cyan]")
    console.print("📊 Expected: 3-5 tool calls, no validation errors")
    console.print("-" * 60)
    
    # Track tool calls
    tool_call_count = 0
    validation_errors = 0
    
    class ToolCallTracker:
        def __init__(self, original_display):
            self.original_display = original_display
            
        def __call__(self, tool_name, output):
            nonlocal tool_call_count, validation_errors
            tool_call_count += 1
            
            # Check for validation errors
            if "validation error" in str(output).lower() or "required property" in str(output).lower():
                validation_errors += 1
                console.print(f"❌ Validation error in {tool_name}")
            
            console.print(f"📤 Tool call #{tool_call_count}: {tool_name}")
            return self.original_display(tool_name, output)
    
    # Run the test
    agent = SSBAgent()
    
    # Monkey patch to track calls
    original_display = agent._display_tool_output
    agent._display_tool_output = ToolCallTracker(original_display)
    
    start_time = time.time()
    
    try:
        result = await asyncio.wait_for(agent.process_query(test_query), timeout=60.0)
        success = True
        end_time = time.time()
        
        console.print(f"\n✅ Test Results:")
        console.print(f"   • Total tool calls: {tool_call_count}")
        console.print(f"   • Validation errors: {validation_errors}")
        console.print(f"   • Duration: {end_time - start_time:.1f}s")
        console.print(f"   • Success: {success}")
        
        # Evaluate improvement
        if tool_call_count <= 5 and validation_errors == 0:
            console.print("\n🎉 [green]OPTIMIZATION SUCCESSFUL![/green]")
            console.print("   Agent is now operating efficiently!")
        elif tool_call_count <= 7:
            console.print("\n⚡ [yellow]PARTIAL IMPROVEMENT[/yellow]")
            console.print(f"   Reduced calls but still {tool_call_count} total")
        else:
            console.print("\n❌ [red]OPTIMIZATION NEEDED[/red]")
            console.print(f"   Still making {tool_call_count} calls")
        
        if len(result) > 100:
            console.print(f"\n📝 Final answer length: {len(result)} characters")
    
    except asyncio.TimeoutError:
        console.print("⏰ Test timed out - agent may still be inefficient")
    except Exception as e:
        console.print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_optimization())