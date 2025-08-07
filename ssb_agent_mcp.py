#!/usr/bin/env python3
"""
Simple SSB Agent MCP - Minimal implementation with Azure OpenAI

Usage:
    python ssb_agent_mcp.py "Your query about Norwegian statistics"
"""

import asyncio
import os
import sys
import logging
import json

import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents import Agent, run, set_default_openai_client, set_default_openai_api, set_tracing_disabled, ItemHelpers
from agents.mcp import MCPServerStdio, MCPServerStdioParams

# Setup
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=False, show_path=False, markup=True)]
)

for handler in logging.root.handlers:
    if isinstance(handler, RichHandler):
        handler.setLevel(logging.INFO)

class SSBAgent:
    """Enhanced SSB Agent using Azure OpenAI and advanced MCP capabilities."""
    
    def __init__(self):
        load_dotenv()
        set_tracing_disabled(True)
        
        # Configure Azure OpenAI client
        azure_client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        set_default_openai_client(azure_client)
        set_default_openai_api("chat_completions")
        
        # Initialize MCP server with correct parameters
        params = MCPServerStdioParams(
            command=sys.executable,
            args=["mcp_server.py"]
        )
        
        self.mcp_server = MCPServerStdio(
            params=params,
            name="SSB Statistics MCP Server",
            cache_tools_list=True
        )
    
    def _display_tool_output(self, tool_name: str, output: any) -> None:
        """Display tool output in human-readable format with intelligent parsing and error detection."""
        console.print(f"[bold cyan]📤 {tool_name} Result:[/bold cyan]")
        
        try:
            # Handle multiple possible output formats from MCP/Agents SDK
            parsed_output = output
            
            # Case 1: String containing JSON with {"type":"text","text":"..."} format
            if isinstance(output, str):
                try:
                    # First parse the outer JSON
                    outer_json = json.loads(output)
                    if isinstance(outer_json, dict) and outer_json.get("type") == "text":
                        # Extract the inner text content and parse it as JSON
                        text_content = outer_json.get("text", "")
                        try:
                            parsed_output = json.loads(text_content)
                        except json.JSONDecodeError:
                            parsed_output = text_content
                    else:
                        parsed_output = outer_json
                except json.JSONDecodeError:
                    parsed_output = output
            
            # Case 2: Dict with "type" and "text" (already parsed outer JSON)
            elif isinstance(output, dict) and "type" in output and output["type"] == "text":
                text_content = output.get("text", "")
                try:
                    parsed_output = json.loads(text_content)
                except json.JSONDecodeError:
                    parsed_output = text_content
            
            # Case 3: Dict with "content" key (MCP format)
            elif isinstance(output, dict) and "content" in output:
                content = output["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Extract from content array
                    first_content = content[0]
                    if isinstance(first_content, dict) and "text" in first_content:
                        try:
                            parsed_output = json.loads(first_content["text"])
                        except json.JSONDecodeError:
                            parsed_output = first_content["text"]
                    else:
                        parsed_output = content
                else:
                    parsed_output = content
            
            # Case 4: Already a dict - use as-is
            elif isinstance(output, dict):
                parsed_output = output
            
            # Case 5: Other types - convert to string
            else:
                parsed_output = str(output)
            
            # Display in human-readable format instead of raw JSON
            if isinstance(parsed_output, dict):
                # Check for errors first
                if "error" in parsed_output:
                    console.print(f"[red]❌ Error: {parsed_output['error']}[/red]")
                    
                    if "details" in parsed_output:
                        details = parsed_output["details"]
                        if isinstance(details, dict):
                            console.print("[yellow]📋 Error Details:[/yellow]")
                            for key, value in details.items():
                                console.print(f"  • {key}: {value}")
                    
                    if "suggestion" in parsed_output:
                        console.print(f"[blue]💡 Suggestion: {parsed_output['suggestion']}[/blue]")
                    
                    return
                
                # Handle successful responses based on tool type
                if tool_name == "search_tables_advanced":
                    self._display_search_results(parsed_output)
                elif tool_name == "analyze_table_structure":
                    self._display_table_analysis(parsed_output)
                elif tool_name == "get_filtered_data":
                    self._display_filtered_data(parsed_output)
                else:
                    # Generic structured display
                    self._display_generic_output(parsed_output)
            else:
                # Simple text output
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
                    t.get("id", ""),
                    t.get("title", "")[:50] + ("..." if len(t.get("title", "")) > 50 else ""),
                    t.get("updated", "")[:10],
                    str(t.get("score", ""))
                )
            
            console.print(table)
    
    def _display_table_analysis(self, data: dict) -> None:
        """Display table structure analysis in human-readable format"""
        table_id = data.get("table_id", "Unknown")
        title = data.get("title", "No title")
        variables = data.get("variables", [])
        aggregation_options = data.get("aggregation_options", {})
        
        console.print(f"[green]📊 Table Analysis: {table_id}[/green]")
        console.print(f"[white]Title: {title}[/white]")
        
        if variables:
            console.print(f"[yellow]📋 Available Dimensions ({len(variables)}):[/yellow]")
            
            var_table = Table()
            var_table.add_column("Dimension", style="cyan")
            var_table.add_column("Label", style="white", max_width=30)
            var_table.add_column("Values", style="dim")
            var_table.add_column("Aggregation", style="green")
            
            for var in variables[:10]:  # Show first 10 dimensions
                agg_status = "✅ Available" if var.get("has_aggregation", False) else "❌ None"
                var_table.add_row(
                    var.get("code", ""),
                    var.get("label", "")[:30],
                    f"{var.get('total_values', 0)} values",
                    agg_status
                )
            
            console.print(var_table)
        
        # Show aggregation options if available
        if aggregation_options:
            console.print(f"[blue]🎯 Aggregation Options Available:[/blue]")
            for dim_name, agg_data in aggregation_options.items():
                console.print(f"[cyan]  {dim_name}:[/cyan]")
                
                # Show valuesets
                if agg_data.get("valuesets"):
                    console.print("    [dim]Valuesets:[/dim]")
                    for vs in agg_data["valuesets"][:3]:  # Show first 3
                        console.print(f"      • {vs['id']} - {vs['label']}")
                
                # Show aggregations
                if agg_data.get("aggregations"):
                    console.print("    [dim]Aggregations:[/dim]")
                    for agg in agg_data["aggregations"][:3]:  # Show first 3
                        console.print(f"      • {agg['id']} - {agg['label']}")
        
        # Show query suggestions if available
        suggestions = data.get("query_suggestions", [])
        if suggestions:
            console.print(f"[blue]💡 Query Suggestions:[/blue]")
            for i, suggestion in enumerate(suggestions[:3], 1):
                console.print(f"  {i}. {suggestion.get('description', 'No description')}")
                if suggestion.get('example'):
                    console.print(f"     [dim]Example: {suggestion['example']}[/dim]")
    
    def _display_filtered_data(self, data: dict) -> None:
        """Display filtered data results in human-readable format"""
        table_id = data.get("table_id", "Unknown")
        
        if "total_data_points" in data or "returned_data_points" in data:
            console.print(f"[green]📈 Data Retrieved from {table_id}[/green]")
            data_points = data.get('total_data_points', data.get('returned_data_points', 0))
            console.print(f"[white]Data Points: {data_points}[/white]")
            
            if "formatted_data" in data and data["formatted_data"]:
                console.print("[yellow]📋 Sample Data:[/yellow]")
                
                # Create a table for the data
                data_table = Table()
                sample_data = data["formatted_data"][:10]  # Show first 10 rows
                
                if sample_data:
                    # Add columns based on first row
                    first_row = sample_data[0]
                    for key in first_row.keys():
                        if key != "value":
                            data_table.add_column(key, style="cyan")
                    data_table.add_column("Value", style="green")
                    
                    # Find max value for highlighting if this looks like comparison data
                    all_values = [row.get("value", 0) for row in data["formatted_data"] if isinstance(row.get("value"), (int, float))]
                    max_value = max(all_values) if all_values else None
                    
                    # Add data rows
                    for row in sample_data:
                        row_data = []
                        for key in first_row.keys():
                            if key != "value":
                                row_data.append(str(row.get(key, "")))
                        
                        # Highlight max value
                        value = row.get("value", "")
                        if max_value and value == max_value:
                            row_data.append(f"[bold green]{value}[/bold green] ⭐")
                        else:
                            row_data.append(str(value))
                        
                        data_table.add_row(*row_data)
                    
                    console.print(data_table)
                    
                    # Show summary for comparison queries
                    if len(data["formatted_data"]) > 10:
                        console.print(f"[dim]... and {len(data['formatted_data']) - 10} more rows[/dim]")
                    
                    # Show max/min summary if numeric data
                    if all_values and len(all_values) > 1:
                        console.print(f"[blue]📊 Summary:[/blue]")
                        console.print(f"  • Highest value: [bold green]{max_value:,}[/bold green]")
                        console.print(f"  • Lowest value: [dim]{min(all_values):,}[/dim]")
                        console.print(f"  • Total entries: {len(all_values)}")
                        
                        # Find and show the highest entry details
                        max_entry = next((row for row in data["formatted_data"] if row.get("value") == max_value), None)
                        if max_entry:
                            console.print(f"  • [bold green]Winner:[/bold green] {max_entry.get('Region', 'Unknown')} with {max_value:,} people")
                        
        else:
            console.print(f"[yellow]⚠️ No data returned from {table_id}[/yellow]")
    
    def _display_generic_output(self, data: dict) -> None:
        """Display generic structured output in human-readable format"""
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                console.print(f"[cyan]{key}:[/cyan] [dim]{type(value).__name__} with {len(value)} items[/dim]")
            else:
                console.print(f"[cyan]{key}:[/cyan] [white]{value}[/white]")
    
    async def process_query(self, query: str) -> str:
        """Process a query using the SSB agent with enhanced reasoning capture and better output formatting."""
        await self.mcp_server.connect()
        
        try:
            # Create agent with enhanced instructions for better table discovery
            agent = Agent(
                name="SSB Statistical Expert",
                instructions="""You are an expert Norwegian statistics analyst that efficiently uses SSB API tools.

🎯 CORE PRINCIPLE: Minimize tool calls - maximum 4 calls per query. Each call must have a clear purpose.

📋 AVAILABLE TOOLS:
1. search_tables_advanced - Find relevant tables (USE EXACTLY ONCE)
2. analyze_table_structure - Get dimensions + aggregation options  
3. discover_dimension_values - Find available codes for specific dimensions
4. discover_code_lists - Find aggregation options (counties, municipalities)
5. get_filtered_data - Retrieve actual data with proper filtering

🚀 OPTIMAL WORKFLOW (3-4 tool calls total):

FOR COMPARISON QUERIES ("which X has most Y"):
1. search_tables_advanced(query) → find best table
2. analyze_table_structure(table_id) → get dimensions + check aggregation_options
3. get_filtered_data with aggregation → get ALL comparison data in ONE call

FOR SPECIFIC DATA QUERIES:
1. search_tables_advanced(query) → find best table  
2. analyze_table_structure(table_id) → get dimensions
3. discover_dimension_values(table_id, dimension) → get specific codes (if needed)
4. get_filtered_data → retrieve data

🔥 CRITICAL SUCCESS RULES:

DIMENSION NAMES:
- analyze_table_structure returns EXACT API dimension names
- NEVER use Norwegian display names in API calls
- Common translations: "region"→"Region", "år"→"Tid", "statistikkvariabel"→"ContentsCode"
- Always use the exact names from analyze_table_structure results

AGGREGATION STRATEGY:
- analyze_table_structure shows "aggregation_options" for each dimension
- For county-level data: use aggregation with "fylker" in the ID + outputValues="single"
- For municipality data: use aggregation with "kommun" in the ID + outputValues="aggregated"
- Example: code_lists={"Region": "agg_Fylker2024"}, output_values={"Region": "single"}

COMPARISON QUERIES (e.g., "which county has most people"):
- Use filters={"Region": "*"} to get ALL regions in ONE call
- Use appropriate aggregation to get county-level data
- NEVER make separate calls for individual regions
- Let the data show all results - don't pre-judge the answer

ERROR PREVENTION:
- If discover_dimension_values fails, read the error - it shows available_dimensions
- Use EXACT dimension names from error messages
- If get_filtered_data fails, filters parameter is REQUIRED and must be a dict

⚡ SPEED OPTIMIZATIONS:
- NEVER call search_tables_advanced more than once
- Skip discover_dimension_values if you can use wildcards (*) 
- For "which has most" queries: go straight to get_filtered_data with "*" wildcard
- Only use discover_code_lists if analyze_table_structure shows aggregation options

🎯 COUNTY COMPARISON EXAMPLE:
Query: "Which county has most people?"
1. search_tables_advanced("befolkning fylke") 
2. analyze_table_structure(best_table) → shows Region dimension with aggregation options
3. get_filtered_data(
     filters={"Region": "*", "ContentsCode": "Folkemengde", "Tid": "2025"},
     code_lists={"Region": "agg_Fylker2024"}, 
     output_values={"Region": "single"}
   ) → Gets ALL counties in one call

✅ SUCCESS INDICATORS:
- Maximum 4 tool calls
- No failed tool calls due to wrong dimension names
- Comparison queries get ALL data in single get_filtered_data call
- Proper use of aggregation for administrative levels

❌ AVOID:
- Multiple search_tables_advanced calls
- Using Norwegian dimension names in API calls
- Multiple get_filtered_data calls for comparison data
- Calling discover_dimension_values when wildcards work
- Ignoring aggregation options for geographic queries

Remember: Efficiency and accuracy over exploration. Use the tools purposefully.""",
                model=os.getenv("AZURE_OPENAI_O3MINI_DEPLOYMENT", "o3-mini"),
                mcp_servers=[self.mcp_server]
            )
            
            # Use streaming to capture reasoning and tool usage with o3-mini
            result = run.Runner.run_streamed(
                agent, 
                query, 
                max_turns=20  # Increase turn limit for complex queries
            )
            
            console.print("🧠 o3-mini Reasoning Model Analysis")
            
            # Enhanced streaming with better reasoning capture
            reasoning_content = []
            tool_calls = []
            final_content = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Processing...", total=None)
                
                async for event in result.stream_events():
                    if event.type == "raw_response_event":
                        # Handle raw response events from o3-mini (includes reasoning)
                        if hasattr(event.data, 'type'):
                            # Capture reasoning events (o3-mini reasoning)
                            if event.data.type == "response.reasoning.delta":
                                if hasattr(event.data, 'delta') and event.data.delta:
                                    reasoning_content.append(event.data.delta)
                                    console.print(f"[bold blue]💭 {event.data.delta.strip()}[/bold blue]")
                            elif event.data.type == "response.reasoning.done":
                                if reasoning_content:
                                    total_chars = len(''.join(reasoning_content))
                                    console.print(f"\n[bold blue]🧠 Reasoning Complete[/bold blue] ({total_chars} chars)\n")
                                else:
                                    console.print("\n[yellow]⚠️ No reasoning content captured - o3-mini reasoning may not be available in this deployment[/yellow]\n")
                            elif event.data.type == "response.reasoning.summary":
                                # Handle reasoning summary if available (limited access feature)
                                if hasattr(event.data, 'summary') and event.data.summary:
                                    console.print(f"\n[bold blue]📝 Reasoning Summary:[/bold blue]")
                                    console.print(f"[blue]{event.data.summary}[/blue]\n")
                            elif event.data.type == "response.output_text.delta":
                                if hasattr(event.data, 'delta') and event.data.delta:
                                    final_content.append(event.data.delta)
                                    console.print(f"[green]{event.data.delta}[/green]", end="")
                            elif event.data.type == "response.output_text.done":
                                console.print("\n")
                    
                    elif event.type == "run_item_stream_event":
                        # Handle higher-level events (tool calls, outputs)
                        if event.item.type == "tool_call_item":
                            # Extract tool name and arguments properly
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
                            console.print(f"\n[yellow]🔧 Calling: {tool_name}[/yellow]")
                            
                            # Display tool parameters in a nice table
                            if tool_args:
                                param_table = Table(show_header=False, box=None, padding=(0, 1))
                                param_table.add_column("Parameter", style="dim")
                                param_table.add_column("Value", style="white")
                                
                                for key, value in tool_args.items():
                                    if isinstance(value, dict):
                                        # Format nested dicts nicely
                                        formatted_value = json.dumps(value, indent=2)
                                        param_table.add_row(key, formatted_value)
                                    elif isinstance(value, list):
                                        param_table.add_row(key, ", ".join(str(v) for v in value))
                                    else:
                                        param_table.add_row(key, str(value))
                                
                                console.print(param_table)
                        
                        elif event.item.type == "tool_call_output_item":
                            # Display tool output in human-readable format
                            tool_output = event.item.output
                            last_tool = tool_calls[-1] if tool_calls else "unknown"
                            

                            self._display_tool_output(last_tool, tool_output)
                        
                        elif event.item.type == "message_output_item":
                            # Final message output
                            pass
                    
                    elif event.type == "agent_updated_stream_event":
                        console.print(f"[blue]🤖 Agent: {event.new_agent.name}[/blue]")
            
            # Get final output
            final_output = result.final_output or ''.join(final_content)
            
            # Display run summary with better formatting
            console.print("\n" + "="*60)
            console.print("🎯 Run Summary")
            console.print(f"🔧 Tools Used: {len(tool_calls)} calls across {len(set(tool_calls))} tools")
            
            if tool_calls:
                # Create summary table of tool usage
                tool_summary = Table(title="Tool Usage Summary")
                tool_summary.add_column("Tool", style="cyan")
                tool_summary.add_column("Calls", style="green")
                tool_summary.add_column("Status", style="green")
                
                tool_counts = {}
                for tool in tool_calls:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
                
                for tool, count in tool_counts.items():
                    tool_summary.add_row(tool, str(count), "✅ Success")
                
                console.print(tool_summary)
            
            # Show reasoning status
            if reasoning_content:
                console.print(f"🧠 Reasoning captured: {len(''.join(reasoning_content))} characters")
            else:
                console.print("🧠 No reasoning trace captured (may not be available for this model/deployment)")
            
            console.print("="*60)
            
            return final_output
            
        finally:
            await self.mcp_server.cleanup()
    
    def _display_run_summary(self, tool_calls: list, reasoning_content: list, tool_outputs: dict) -> None:
        """Display a comprehensive summary of the agent run."""
        console.print("\n" + "="*60)
        console.print("[bold blue]🎯 Run Summary[/bold blue]")
        
        # Tool usage summary
        if tool_calls:
            unique_tools = list(set(tool_calls))
            console.print(f"[bold yellow]🔧 Tools Used:[/bold yellow] {len(tool_calls)} calls across {len(unique_tools)} tools")
            
            tool_table = Table(show_header=True, header_style="bold magenta")
            tool_table.add_column("Tool", style="cyan")
            tool_table.add_column("Calls", justify="right", style="yellow")
            tool_table.add_column("Status", style="green")
            
            for tool in unique_tools:
                call_count = tool_calls.count(tool)
                status = "✅ Success" if tool in tool_outputs else "⏳ Running"
                if tool in tool_outputs and isinstance(tool_outputs[tool], dict) and "error" in str(tool_outputs[tool]):
                    status = "❌ Error"
                tool_table.add_row(tool, str(call_count), status)
            
            console.print(tool_table)
        
        # Reasoning summary
        if reasoning_content:
            total_reasoning = ''.join(reasoning_content)
            console.print(f"[bold blue]🧠 Reasoning:[/bold blue] {len(total_reasoning)} characters of internal thought process")
            
            # Show a sample of reasoning if available
            if len(total_reasoning) > 100:
                sample = total_reasoning[:200] + "..." if len(total_reasoning) > 200 else total_reasoning
                console.print(f"[dim italic]Sample: {sample}[/dim italic]")
        else:
            console.print("[dim]🧠 No reasoning trace captured (may not be available for this model/deployment)[/dim]")
        
        console.print("="*60 + "\n")

async def main():
    """Main CLI interface."""
    
    if len(sys.argv) != 2:
        console.print("[red]Usage:[/red] python ssb_agent_mcp.py \"Your query\"")
        console.print("\n[bold]Examples:[/bold]")
        console.print("\n[cyan]Basic Queries:[/cyan]")
        console.print("  python ssb_agent_mcp.py \"befolkning\"")
        console.print("  python ssb_agent_mcp.py \"arbeidsledighet\"")
        console.print("  python ssb_agent_mcp.py \"utdanning\"")
        console.print("\n[cyan]Analysis Queries:[/cyan]")
        console.print("  python ssb_agent_mcp.py \"Compare population between Oslo and Bergen\"")
        console.print("  python ssb_agent_mcp.py \"Historical unemployment trends since 2020\"")
        console.print("  python ssb_agent_mcp.py \"Latest education statistics by region\"")
        console.print("\n[cyan]English Queries:[/cyan]")
        console.print("  python ssb_agent_mcp.py \"What are the latest population statistics?\"")
        console.print("  python ssb_agent_mcp.py \"Show me income data by age group\"")
        console.print("\n[dim]💡 Tip: Norwegian keywords often yield better results[/dim]")
        sys.exit(1)
    
    query = sys.argv[1]
    
    console.print(Panel(
        Text(query, style="bold cyan"),
        title="[bold blue]🤖 SSB Agent Query[/bold blue]",
        border_style="blue"
    ))
    
    console.print("[dim]Connecting to enhanced MCP server with advanced SSB API capabilities...[/dim]")
    
    agent = SSBAgent()
    
    console.print("[dim]Starting analysis with o3-mini reasoning model...[/dim]\n")
    
    # Process query with streaming to show reasoning
    answer = await agent.process_query(query)
    
    # Only show final answer panel if we have content
    if answer and answer.strip():
        console.print()
        console.print(Panel(
            Text(answer, style="white"),
            title="[bold green]📋 Final Answer[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))

if __name__ == "__main__":
    asyncio.run(main())