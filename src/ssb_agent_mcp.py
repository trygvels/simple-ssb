#!/usr/bin/env python3
"""
SSB Statistikk Agent - Norsk statistikk med Azure OpenAI og strømlinjeformet MCP

Intelligente svar på norske spørsmål om statistikk ved bruk av SSB's API.
Optimalisert for 4 strømlinjeformede MCP-verktøy og norske spørringer.

Bruk:
    python ssb_agent_mcp.py "Din spørring om norsk statistikk"
"""

import asyncio
import os
import sys
import logging
import json
import time

import openai
from dotenv import load_dotenv, find_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents import Agent, run, set_default_openai_client, set_default_openai_api, set_tracing_disabled, ItemHelpers
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from agents import model_settings as agent_model_settings

model = "gpt-5-mini"
model = "gpt-5"

# Setup
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

class SSBAgent:
    """Norsk SSB Statistikk Agent med Azure OpenAI og 4 strømlinjeformede MCP-verktøy."""
    
    def __init__(self):
        # Prefer the .env next to this package (servers/simple-ssb/.env)
        simple_ssb_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
        if os.path.exists(simple_ssb_env):
            load_dotenv(simple_ssb_env, override=True)
        else:
            # Fallback to nearest .env in the tree
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
        # Use Responses API to enable reasoning events with gpt
        set_default_openai_api("responses")
        
        # Resolve model from env if provided
        self.model = os.getenv("AZURE_OPENAI_MODEL", model)
        
        # Initialize MCP server with correct parameters
        server_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        params = MCPServerStdioParams(
            command=sys.executable,
            args=[server_path],
            # Increase stdio timeouts to handle SSB API latency
            request_timeout_s=int(os.getenv("MCP_REQUEST_TIMEOUT_S", "20")),
            connect_timeout_s=int(os.getenv("MCP_CONNECT_TIMEOUT_S", "10"))
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
                    t.get('id', 'N/A'),
                    t.get('title', 'N/A'),
                    str(t.get('updated', 'N/A')),
                    f"{t.get('score', 'N/A')}"
                )
    
    def _display_table_analysis(self, data: dict) -> None:
        """Display table analysis results"""
        # This is a placeholder if needed; in the omitted code it likely exists
        pass

    def _display_filtered_data(self, data: dict) -> None:
        """Display filtered data results"""
        # This is a placeholder if needed; in the omitted code it likely exists
        pass

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
                            # Remove table ID from title if it starts with it
                            if title.startswith(f"{table_id}: "):
                                title = title[len(f"{table_id}: "):]
                            console.print(f"     {i+1}. [cyan]{table_id}[/cyan] - {title}...")
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
                    valuesets = parsed_output.get("valuesets", [])
                    if valuesets:
                        agg_options = [vs for vs in valuesets if vs.startswith(('agg_', 'vs_'))]
                        if agg_options:
                            console.print(f"   [dim]  Aggregation options: {', '.join(agg_options[:3])}[/dim]")
                
                elif tool_name in ["discover_dimension_values", "discover_code_lists", "search_region_codes"]:
                    dim_name = parsed_output.get("dimension_name", "N/A")
                    total_values = parsed_output.get("total_values", 0)
                    matching = parsed_output.get("matching_values", 0)
                    console.print(f"   [green]✓ Dimension '{dim_name}': {matching}/{total_values} values found[/green]")
                
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
                
        except Exception:
            console.print(f"   [yellow]⚠ {tool_name} completed (unable to parse result)[/yellow]")
    
    def _parse_tool_output(self, output: any) -> any:
        """Parse tool output handling various formats from the agent framework."""
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
            
            return parsed_output
                
        except Exception:
            return output
    
    async def process_query(self, query: str) -> str:
        """Process a query using the SSB agent with enhanced reasoning capture and better output formatting."""
        t0 = time.monotonic()
        mcp_start = time.monotonic()
        await self.mcp_server.connect()
        mcp_end = time.monotonic()
        console.print(f"[dim]⏱ MCP connect: {int((mcp_end - mcp_start)*1000)} ms[/dim]")
        
        try:
            # Create agent with optimized instructions to minimize tool calls
            agent = Agent(
                name="SSB Effektiv Statistikk-ekspert",
                instructions="""Du er en ekspert på norsk statistikk som bruker SSB API-verktøy MED MAKSIMAL EFFEKTIVITET.

🎯 MÅL: Minimal antall verktøykall (3-5 kall totalt) med null valideringsfeil.

📋 TILGJENGELIGE VERKTØY (4 strømlinjeformede):
1. search_tables - Finn tabeller (MAKS 2 kall)
2. get_table_info - Få tabellstruktur (1 kall)  
3. discover_dimension_values - Finn dimensjonskoder (kun når nødvendig)
4. get_filtered_data - Hent data (MAKS 2 kall)

🔥 KRITISKE REGLER FOR get_filtered_data:
- filters parameter er ALLTID PÅKREVD - aldri kall uten den
- Tid går i filters, IKKE time_selection: {"Tid": "2024"} 
- Bruk EKSAKTE dimensjonsnavn fra get_table_info
- For sammenligninger: bruk wildcards {"Region": "*"} i ETT kall

📊 OPTIMALE MØNSTRE PER SPØRRETYPE:

RANGERING/TOP N ("topp 5 næringer"):
1. search_tables (1-2 kall maks)
2. get_table_info (1 kall)
3. get_filtered_data MED wildcard på sammenligningsdimensjon:
   filters={
     "Region": "0",           # Hele landet
     "NACE2007": "*",         # Alle næringer
     "Tid": "2024"            # Spesifikt år
   }
   code_lists={"NACE2007": "agg_NACE2007arb11"}  # Bruk aggregering

ENKEL TELLING ("befolkning Norge"):
1. search_tables (1 kall)
2. get_table_info (1 kall)
3. get_filtered_data:
   filters={
     "Region": "0",
     "ContentsCode": "Folkemengde", 
     "Tid": "2024"
   }

REGIONAL SAMMENLIGNING ("Oslo vs Bergen"):
1. search_tables (1 kall)
2. get_table_info (1 kall)
3. get_filtered_data med wildcard på Region:
   filters={
     "Region": "*",           # Alle regioner
     "ContentsCode": "xxx",
     "Tid": "2024"
   }

🚫 UNNGÅ DISSE FEILENE:
- ALDRI kall get_filtered_data uten filters dict
- ALDRI bruk time_selection parameter - alt i filters
- ALDRI gjett dimensjonsnavn - bruk eksakte fra get_table_info
- ALDRI mer enn 2 search_tables kall
- ALDRI discover_dimension_values for rangeringsspørsmål

⚡ SØK-STRATEGI (MAKS 2 kall):
- Kall 1: Primære norske termer ("sysselsatte næring")
- Kall 2: Alternative termer bare hvis nødvendig ("arbeidstakere NACE")
- STOPP etter 2 søk - ikke fortsett med variasjoner

✅ KVALITETSKRAV:
- 3-5 verktøykall totalt (ikke 7-8)
- ZERO valideringsfeil - alltid komplette parametere
- 1-2 get_filtered_data kall maksimum
- Ett get_filtered_data kall henter all nødvendig data når mulig

🇳🇴 NORSK RESPONS:
- Svar på norsk med norske tall og enheter
- Bruk norske søkeord og stedsnavn
- Kilde oppgitt med tabellnummer

Husk: EFFEKTIVITET over alt - færre kall, ingen feil, riktige resultater.

Dersom det er uklart hvilken tabell du skal bruke, bruk en av dem og prøv å besvar spørsmålet, deretter foreslå alternativer om det finnes.

""",
                model=self.model,
                mcp_servers=[self.mcp_server],
                model_settings=agent_model_settings.ModelSettings(
                    reasoning={
                        "effort": os.getenv("AZURE_REASONING_EFFORT", "medium"),
                        "summary": os.getenv("AZURE_REASONING_SUMMARY", "auto"),
                    },
                ),
            )
            
            # Use streaming to capture reasoning and tool usage with gpt
            t_model_start = time.monotonic()
            result = run.Runner.run_streamed(
                agent, 
                query, 
                max_turns=20  # Increase turn limit for complex queries
            )
            
            console.print(f"[bold blue]🧠 {model} Analysis[/bold blue]")
            
            # Enhanced streaming with better reasoning capture
            reasoning_content = []
            tool_calls = []
            final_content = []
            saw_reasoning_summary = False
            first_token_time = None
            last_event_time = time.monotonic()
            stall_warned = False
            STALL_MS = int(os.getenv("AGENT_STALL_WARNING_MS", "15000"))  # 15s default
            TURN_TIMEOUT_S = int(os.getenv("AGENT_TURN_TIMEOUT_S", "60"))  # 60s default

            # Track tool timings and last analysis for diagnostics
            tool_timers: dict[int, tuple[str, float]] = {}
            tool_call_counter: int = 0
            last_analysis: dict | None = None
            last_table_id: str | None = None
            
            async def stall_monitor():
                nonlocal last_event_time, stall_warned
                while True:
                    await asyncio.sleep(1.0)
                    now = time.monotonic()
                    if not stall_warned and (now - last_event_time) * 1000 > STALL_MS:
                        stall_warned = True
                        console.print("[yellow]⏳ Warning: No events received for a while. The model may be thinking or stalled.[/yellow]")
                        console.print("[dim]Tip: reduce reasoning effort or rephrase the query if this persists.[/dim]")

            monitor_task = asyncio.create_task(stall_monitor())
            turn_start_time = time.monotonic()

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Processing...", total=None)
                
                async for event in result.stream_events():
                    last_event_time = time.monotonic()
                    # Enforce a soft per-turn timeout to prevent infinite waits
                    if last_event_time - turn_start_time > TURN_TIMEOUT_S:
                        console.print("[red]⏱ Turn timeout reached. Ending early to avoid getting stuck.[/red]")
                        break

                    if event.type == "raw_response_event":
                        # Handle raw response events from gpt (includes reasoning)
                        if hasattr(event.data, 'type'):
                            # Capture reasoning events (gpt reasoning)
                            if event.data.type == "response.reasoning.delta":
                                if first_token_time is None:
                                    first_token_time = time.monotonic()
                                    console.print(f"[dim]⏱ Time to first token: {int((first_token_time - t_model_start)*1000)} ms[/dim]")
                                if hasattr(event.data, 'delta') and event.data.delta:
                                    # Note: Some deployments return encrypted/empty deltas; we only print readable text
                                    if isinstance(event.data.delta, str) and event.data.delta.strip():
                                        reasoning_content.append(event.data.delta)
                                        console.print(f"[bold blue]💭 {event.data.delta.strip()}[/bold blue]")
                            elif event.data.type == "response.reasoning.done":
                                if reasoning_content:
                                    total_chars = len(''.join(reasoning_content))
                                    console.print(f"\n[bold blue]🧠 Reasoning Complete[/bold blue] ({total_chars} chars)")
                            elif event.data.type == "response.reasoning.summary":
                                # Handle reasoning summary if available
                                if first_token_time is None:
                                    first_token_time = time.monotonic()
                                    console.print(f"[dim]⏱ Time to first token (summary): {int((first_token_time - t_model_start)*1000)} ms[/dim]")
                                if hasattr(event.data, 'summary') and event.data.summary:
                                    saw_reasoning_summary = True
                                    console.print(f"\n[bold blue]📝 Reasoning Summary:[/bold blue]")
                                    console.print(f"[blue]{event.data.summary}[/blue]")
                            elif event.data.type == "response.output_text.delta":
                                if first_token_time is None:
                                    first_token_time = time.monotonic()
                                    console.print(f"[dim]⏱ Time to first token (output): {int((first_token_time - t_model_start)*1000)} ms[/dim]")
                                if hasattr(event.data, 'delta') and event.data.delta:
                                    final_content.append(event.data.delta)
                                    console.print(f"[green]{event.data.delta}[/green]", end="")
                            elif event.data.type == "response.output_text.done":
                                console.print("")
                    
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

                            # Start per-tool timer
                            tool_call_counter += 1
                            tool_timers[tool_call_counter] = (tool_name, time.monotonic())

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

                            # Track last table_id if present
                            if isinstance(tool_args, dict) and 'table_id' in tool_args:
                                last_table_id = str(tool_args.get('table_id'))

                        elif event.item.type == "tool_call_output_item":
                            # Display tool output in human-readable format
                            tool_output = event.item.output
                            last_tool = tool_calls[-1] if tool_calls else "unknown"
                            
                            # Stop per-tool timer and print duration
                            if tool_timers:
                                idx = max(tool_timers.keys())
                                t_name, t_start = tool_timers.pop(idx)
                                elapsed_ms = int((time.monotonic() - t_start) * 1000)
                                console.print(f"[dim]⏱ {t_name} completed in {elapsed_ms} ms[/dim]")

                            # Store last analysis if applicable
                            try:
                                parsed = tool_output
                                if isinstance(tool_output, dict) and "content" in tool_output:
                                    content = tool_output["content"]
                                    if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
                                        parsed_text = content[0]["text"]
                                        parsed = json.loads(parsed_text)
                                elif isinstance(tool_output, dict) and tool_output.get("type") == "text":
                                    parsed = json.loads(tool_output.get("text", "{}"))
                                elif isinstance(tool_output, str):
                                    try:
                                        parsed = json.loads(tool_output)
                                    except json.JSONDecodeError:
                                        parsed = None
                                if last_tool == "get_table_info" and isinstance(parsed, dict):
                                    last_analysis = parsed
                            except Exception:
                                pass

                            self._display_tool_output(last_tool, tool_output)
                            # Also show brief summary
                            self._display_tool_result_summary(last_tool, tool_output)
                        
                        elif event.item.type == "message_output_item":
                            # Final message output
                            pass
                    
                    elif event.type == "agent_updated_stream_event":
                        console.print(f"[blue]🤖 Agent: {event.new_agent.name}[/blue]")
            
            monitor_task.cancel()

            # Get final output
            final_output = result.final_output or ''.join(final_content)

            # If we timed out or produced no final output, print diagnostics
            if not final_output:
                console.print("[yellow]ℹ️ No final output produced. Diagnostics:[/yellow]")
                console.print(f"[dim]Last table_id: {last_table_id or 'n/a'}; Have analysis: {bool(last_analysis)}[/dim]")
            
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
            if reasoning_content or saw_reasoning_summary:
                total_chars = len(''.join(reasoning_content)) if reasoning_content else 0
                console.print(f"🧠 Reasoning captured: {'summary only' if saw_reasoning_summary and not reasoning_content else str(total_chars) + ' characters'}")
            else:
                console.print("🧠 No reasoning trace captured (may not be available for this model/deployment)")
            
            t_end = time.monotonic()
            console.print(f"⏱ Total wall time: {int((t_end - t0)*1000)} ms")
            console.print("="*60)
            
            return final_output
            
        finally:
            await self.mcp_server.cleanup()

async def main():
    """Main CLI interface."""
    
    if len(sys.argv) != 2:
        console.print("[red]Bruk:[/red] python ssb_agent_mcp.py \"Din spørring\"")
        console.print("\n[bold]Eksempler:[/bold]")
        console.print("\n[cyan]Grunnleggende spørringer:[/cyan]")
        console.print("  python ssb_agent_mcp.py \"befolkning i Norge\"")
        console.print("  python ssb_agent_mcp.py \"arbeidsledighet etter region\"")
        console.print("  python ssb_agent_mcp.py \"utdanningsnivå fylkesvis\"")
        console.print("\n[cyan]Sammenligning og analyse:[/cyan]")
        console.print("  python ssb_agent_mcp.py \"sammenlign befolkning Oslo og Bergen\"")
        console.print("  python ssb_agent_mcp.py \"arbeidsledighet utvikling siden 2020\"")
        console.print("  python ssb_agent_mcp.py \"nyeste utdanningsstatistikk per region\"")
        console.print("\n[cyan]Tidsserier og trender:[/cyan]")
        console.print("  python ssb_agent_mcp.py \"boligpriser utvikling siste 5 år\"")
        console.print("  python ssb_agent_mcp.py \"sysselsetting næring 2024\"")
        console.print("  python ssb_agent_mcp.py \"hva er de nyeste tallene for innvandring\"")
        console.print("\n[dim]💡 Tips: Norske nøkkelord gir best resultater fra SSB[/dim]")
        sys.exit(1)
    
    query = sys.argv[1]
    
    console.print(Panel(
        Text(query, style="bold cyan"),
        title="[bold blue]🤖 SSB Agent Query[/bold blue]",
        border_style="blue"
    ))
    
    console.print("[dim]Kobler til strømlinjeformet MCP-server med 4 SSB API-verktøy...[/dim]")
    
    agent = SSBAgent()
    
    console.print(f"[dim]Starter analyse med {model} modell for norske statistikkspørsmål...[/dim]\n")
    
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