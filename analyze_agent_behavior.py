#!/usr/bin/env python3
"""
Agent Behavior Analysis Script

Runs the SSB agent with test queries and captures detailed logs for analysis.
Analyzes tool usage patterns, inefficiencies, and optimization opportunities.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ssb_agent_mcp import SSBAgent
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class AgentBehaviorAnalyzer:
    """Analyzes agent behavior patterns and identifies optimization opportunities."""
    
    def __init__(self):
        self.analysis_results = []
        self.logs_dir = Path("tests/logs/behavior_analysis")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    async def analyze_query(self, query: str, expected_tools: List[str] = None) -> Dict[str, Any]:
        """Run a single query and analyze the tool usage pattern."""
        console.print(f"\n🔍 Analyzing query: [cyan]{query}[/cyan]")
        
        # Capture start time
        start_time = time.time()
        
        # Create agent and run query
        agent = SSBAgent()
        
        # Monkey patch to capture tool calls
        tool_calls = []
        original_display = agent._display_tool_output
        
        def capture_tool_calls(tool_name: str, output: Any):
            tool_calls.append({
                "tool_name": tool_name,
                "output": output,
                "timestamp": time.time()
            })
            return original_display(tool_name, output)
        
        agent._display_tool_output = capture_tool_calls
        
        try:
            result = await asyncio.wait_for(agent.process_query(query), timeout=60.0)
            success = True
            error = None
        except asyncio.TimeoutError:
            result = "Query timed out"
            success = False
            error = "timeout"
        except Exception as e:
            result = f"Error: {str(e)}"
            success = False
            error = str(e)
        
        end_time = time.time()
        
        # Analyze tool usage patterns
        analysis = self._analyze_tool_pattern(query, tool_calls, success, end_time - start_time)
        
        # Save detailed log
        log_file = self.logs_dir / f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                "query": query,
                "tool_calls": tool_calls,
                "result": result,
                "analysis": analysis,
                "success": success,
                "error": error
            }, f, indent=2, ensure_ascii=False, default=str)
        
        return analysis
    
    def _analyze_tool_pattern(self, query: str, tool_calls: List[Dict], success: bool, duration: float) -> Dict[str, Any]:
        """Analyze the tool usage pattern for inefficiencies."""
        
        # Count tool usage
        tool_counts = {}
        tool_sequence = []
        errors = []
        
        for call in tool_calls:
            tool_name = call["tool_name"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            tool_sequence.append(tool_name)
            
            # Check for errors in output
            if isinstance(call["output"], dict):
                output_str = str(call["output"])
                if "error" in call["output"] or "Error" in output_str or "validation error" in output_str.lower():
                    errors.append({
                        "tool": tool_name,
                        "error": call["output"]
                    })
        
        # Identify inefficiencies
        inefficiencies = []
        
        # 1. Multiple get_filtered_data calls
        if tool_counts.get("get_filtered_data", 0) > 2:
            inefficiencies.append({
                "type": "excessive_data_calls",
                "description": f"Made {tool_counts['get_filtered_data']} get_filtered_data calls",
                "impact": "high",
                "suggestion": "Use wildcards or aggregation to get all data in 1-2 calls"
            })
        
        # 2. get_filtered_data errors
        data_errors = [e for e in errors if e["tool"] == "get_filtered_data"]
        if data_errors:
            inefficiencies.append({
                "type": "filtering_errors",
                "description": f"Failed get_filtered_data calls: {len(data_errors)}",
                "impact": "medium",
                "suggestion": "Always provide required filters dict, use exact dimension names from get_table_info"
            })
        
        # 3. Excessive search calls
        if tool_counts.get("search_tables", 0) > 3:
            inefficiencies.append({
                "type": "excessive_searches",
                "description": f"Made {tool_counts['search_tables']} search calls",
                "impact": "low",
                "suggestion": "Limit to 2-3 varied search terms"
            })
        
        # 4. discover_dimension_values when wildcard would work
        if "discover_dimension_values" in tool_counts and "get_filtered_data" in tool_counts:
            # Check if wildcard was used effectively
            inefficiencies.append({
                "type": "unnecessary_dimension_discovery",
                "description": "Used discover_dimension_values before wildcard filtering",
                "impact": "low",
                "suggestion": "Try wildcards (*) first for comparison queries"
            })
        
        # Calculate efficiency score
        total_calls = len(tool_calls)
        error_calls = len(errors)
        efficiency_score = max(0, 10 - (total_calls - 4) - (error_calls * 2))  # Optimal is ~4 calls
        
        return {
            "query": query,
            "total_tool_calls": total_calls,
            "tool_counts": tool_counts,
            "tool_sequence": tool_sequence,
            "errors": errors,
            "error_count": len(errors),
            "inefficiencies": inefficiencies,
            "efficiency_score": efficiency_score,
            "duration_seconds": duration,
            "success": success,
            "optimal_sequence": self._suggest_optimal_sequence(query, tool_counts)
        }
    
    def _suggest_optimal_sequence(self, query: str, tool_counts: Dict) -> List[str]:
        """Suggest optimal tool sequence for this type of query."""
        
        if "top" in query.lower() or "sammenlign" in query.lower() or "flest" in query.lower():
            # Comparison/ranking query
            return [
                "search_tables (1-2 calls with Norwegian terms)",
                "get_table_info (1 call on best table)",
                "get_filtered_data (1 call with wildcard on comparison dimension)"
            ]
        elif any(word in query.lower() for word in ["hvor mange", "antall", "statistikk"]):
            # Count/statistics query  
            return [
                "search_tables (1-2 calls)",
                "get_table_info (1 call)",
                "discover_dimension_values (if specific codes needed)",
                "get_filtered_data (1 call with proper filters)"
            ]
        else:
            # General query
            return [
                "search_tables (1-2 calls)",
                "get_table_info (1 call)",
                "get_filtered_data (1-2 calls maximum)"
            ]

async def main():
    """Run behavior analysis on representative queries."""
    
    console.print("\n🤖 SSB Agent Behavior Analysis")
    console.print("=" * 60)
    
    analyzer = AgentBehaviorAnalyzer()
    
    # Test queries representing different patterns
    test_queries = [
        # Comparison/ranking queries (your example)
        "Hvilken næring har flest sysselsatte? Gi meg top 5 i 2024",
        
        # Simple statistics queries  
        "befolkning Norge 2024",
        
        # Regional comparison
        "sammenlign arbeidsledighet mellom Oslo og Bergen",
        
        # Time series  
        "utvikling sysselsetting siden 2020",
        
        # Count query
        "hvor mange jobber i helse og sosial 2024"
    ]
    
    all_analyses = []
    
    for query in test_queries:
        try:
            analysis = await analyzer.analyze_query(query)
            all_analyses.append(analysis)
            
            # Display immediate results
            console.print(f"\n📊 Analysis Results for: [cyan]{query}[/cyan]")
            
            # Create results table
            results_table = Table()
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="white")
            
            results_table.add_row("Total Tool Calls", str(analysis["total_tool_calls"]))
            results_table.add_row("Errors", str(analysis["error_count"]))
            results_table.add_row("Efficiency Score", f"{analysis['efficiency_score']}/10")
            results_table.add_row("Duration", f"{analysis['duration_seconds']:.1f}s")
            
            console.print(results_table)
            
            # Show inefficiencies
            if analysis["inefficiencies"]:
                console.print("\n⚠️ Inefficiencies Found:")
                for ineff in analysis["inefficiencies"]:
                    console.print(f"  • {ineff['type']}: {ineff['description']}")
                    console.print(f"    💡 {ineff['suggestion']}")
            
            # Show optimal sequence
            console.print("\n✅ Suggested Optimal Sequence:")
            for step in analysis["optimal_sequence"]:
                console.print(f"  • {step}")
            
            console.print("\n" + "-" * 60)
            
        except Exception as e:
            console.print(f"❌ Failed to analyze query '{query}': {e}")
    
    # Generate summary report
    if all_analyses:
        console.print("\n📋 SUMMARY REPORT")
        console.print("=" * 60)
        
        avg_calls = sum(a["total_tool_calls"] for a in all_analyses) / len(all_analyses)
        avg_errors = sum(a["error_count"] for a in all_analyses) / len(all_analyses) 
        avg_efficiency = sum(a["efficiency_score"] for a in all_analyses) / len(all_analyses)
        
        summary_table = Table(title="Agent Performance Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Average", style="white")
        summary_table.add_column("Target", style="green")
        
        summary_table.add_row("Tool Calls per Query", f"{avg_calls:.1f}", "4-5")
        summary_table.add_row("Errors per Query", f"{avg_errors:.1f}", "0-1")  
        summary_table.add_row("Efficiency Score", f"{avg_efficiency:.1f}/10", "8-10")
        
        console.print(summary_table)
        
        # Common inefficiency patterns
        all_inefficiencies = []
        for analysis in all_analyses:
            all_inefficiencies.extend(analysis["inefficiencies"])
        
        inefficiency_counts = {}
        for ineff in all_inefficiencies:
            ineff_type = ineff["type"]
            inefficiency_counts[ineff_type] = inefficiency_counts.get(ineff_type, 0) + 1
        
        if inefficiency_counts:
            console.print("\n🎯 Most Common Inefficiency Patterns:")
            for ineff_type, count in sorted(inefficiency_counts.items(), key=lambda x: x[1], reverse=True):
                console.print(f"  • {ineff_type}: {count} occurrences")
        
        console.print(f"\n📁 Detailed logs saved to: {analyzer.logs_dir}")

if __name__ == "__main__":
    asyncio.run(main())