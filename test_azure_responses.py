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

    # Create a simple Responses API request
    response = client.responses.create(
        model=model,
        input=prompt,
    )

    # Try to print friendly text output if available
    text = getattr(response, "output_text", None)
    if text:
        print(text)
        return

    # Fallback: try to extract text from output content
    try:
        outputs = getattr(response, "output", [])
        chunks = []
        for out in outputs or []:
            for content in getattr(out, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    chunks.append(getattr(content, "text", ""))
        if chunks:
            print("\n".join(chunks))
            return
    except Exception:
        pass

    # Last resort: print the raw object
    print(response)


if __name__ == "__main__":
    main() 