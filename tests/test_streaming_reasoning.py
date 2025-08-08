import os
import sys
import asyncio
from contextlib import asynccontextmanager
from rich.console import Console
from agents import Agent, run, set_default_openai_client, set_default_openai_api
import openai

console = Console()

@asynccontextmanager
async def setup_agent(model: str):
    client = openai.AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    set_default_openai_client(client)
    set_default_openai_api("responses")
    agent = Agent(name="Test Agent", instructions="You are concise.", model=model)
    yield agent

async def run_once_and_capture(agent: Agent, prompt: str):
    result = run.Runner.run_streamed(agent, prompt, max_turns=1)
    saw_reasoning = False
    saw_output = False
    async for event in result.stream_events():
        if event.type == "raw_response_event" and hasattr(event.data, "type"):
            if event.data.type == "response.reasoning.summary" and getattr(event.data, "summary", None):
                saw_reasoning = True
            if event.data.type == "response.output_text.delta" and getattr(event.data, "delta", None):
                saw_output = True
    return saw_reasoning, saw_output

async def main():
    model = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-mini")
    async with setup_agent(model) as agent:
        saw_reasoning, saw_output = await run_once_and_capture(agent, "Say hello and think briefly.")
        print(f"reasoning_summary={saw_reasoning} output_streamed={saw_output}")
        # Soft assertions: print booleans so CI/logs can verify

if __name__ == "__main__":
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Skip: missing AZURE_OPENAI_ENDPOINT")
        sys.exit(0)
    asyncio.run(main()) 