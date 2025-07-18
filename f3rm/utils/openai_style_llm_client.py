"""
General OpenAI-style LLM chat completions client using the official openai Python client.

Requires: pip install openai
"""
import os
import openai
import json


def chat_completion(model, messages, base_url=None, api_key=None, **kwargs):
    """
    Send a chat completion request to an OpenAI-compatible LLM server using the openai Python client.

    Args:
        model (str): Model name.
        messages (list): List of message dicts (role/content).
        base_url (str): Base URL of the API endpoint (default: OpenAI).
        api_key (str): API key for Authorization header. Defaults to OPENAI_API_KEY env var.
        **kwargs: Additional fields for the payload (e.g., temperature, max_tokens).

    Returns:
        openai.types.ChatCompletion: The response object from the OpenAI client.
    """
    if base_url is None:
        base_url = "https://api.openai.com/v1"
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("api_key must be provided or set in the OPENAI_API_KEY environment variable.")
    # Initialize the OpenAI client with the provided configuration
    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Create and send the chat completion request
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return response


if __name__ == "__main__":
    # Load server configurations from servers.json
    servers_json_path = "servers.json"
    with open(servers_json_path, 'r') as f:
        servers = json.load(f)
    # Available options: openai-gpt4o, internvl3, llama3.1, llama3.2V
    SELECT = "openai-gpt4o"  # Change this to switch server
    selected_server = servers[SELECT]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hey buddy! How you doin?"},
    ]
    print(f"--- {SELECT} ---")
    try:
        response = chat_completion(
            model=selected_server["model"],
            messages=messages,
            base_url=selected_server["base_url"],
            api_key=selected_server["api_key"],
            temperature=0.7,
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
