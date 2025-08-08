import os
import sys
from openai import AzureOpenAI


def main():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    model = os.getenv("AZURE_OPENAI_MODEL", "gpt-5")

    if not endpoint or not api_key:
        print("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in environment.")
        print("Export your credentials and re-run, e.g.:")
        print("  export AZURE_OPENAI_ENDPOINT=...\n  export AZURE_OPENAI_API_KEY=...\n  export AZURE_OPENAI_API_VERSION=2025-04-01-preview")
        sys.exit(1)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello from the Azure OpenAI Responses API using gpt-5."

    reasoning_effort = os.getenv("AZURE_REASONING_EFFORT", "low")
    reasoning_summary_pref = os.getenv("AZURE_REASONING_SUMMARY", "auto")

    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": reasoning_effort, "summary": reasoning_summary_pref},
    )

    try:
        reasoning_obj = getattr(response, "reasoning", None)
        if reasoning_obj:
            summary = getattr(reasoning_obj, "summary", None)
            if summary:
                print("[Thoughts]\n" + summary + "\n")
    except Exception:
        pass

    text = getattr(response, "output_text", None)
    if text:
        print(text)
        _print_reasoning_usage(response)
        return

    try:
        outputs = getattr(response, "output", [])
        chunks = []
        for out in outputs or []:
            for content in getattr(out, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    chunks.append(getattr(content, "text", ""))
        if chunks:
            print("\n".join(chunks))
            _print_reasoning_usage(response)
            return
    except Exception:
        pass

    print(response)
    _print_reasoning_usage(response)


def _print_reasoning_usage(response) -> None:
    try:
        usage = getattr(response, "usage", None)
        if usage:
            details = getattr(usage, "output_tokens_details", None)
            if details:
                r_tokens = getattr(details, "reasoning_tokens", None)
                if r_tokens is not None:
                    print(f"\n[Reasoning tokens]: {r_tokens}")
    except Exception:
        pass


if __name__ == "__main__":
    main() 