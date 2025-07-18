"""
General OpenAI-style multimodal (text + image) chat completions client using the official openai Python client.

- Accepts messages as a direct argument (user supplies the message dicts, including text and image content, in OpenAI format).
- Provides utility functions for handling image inputs: web URL, file path (prepends file://), and PIL.Image (converts to base64 data URL).
- Ignores audio and video.
- Uses the OpenAI Python client, with api_key defaulting to env var if not provided.
- See: https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#image-inputs

Note: For local file paths, the vLLM server must be launched with --allowed-local-media-path.
"""
import os
import base64
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from io import BytesIO
import openai
from PIL import Image
import requests
import json


def image_url_payload(url: str) -> Dict[str, Any]:
    """Return OpenAI-format image_url content for a web URL."""
    return {"type": "image_url", "image_url": {"url": url}}


def image_file_payload(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Return OpenAI-format image_url content for a local file path (prepends file://)."""
    path = Path(file_path).absolute()
    return {"type": "image_url", "image_url": {"url": f"file://{path}"}}


def image_pil_payload(img: Image.Image, mime_type: str = "image/png") -> Dict[str, Any]:
    """Return OpenAI-format image_url content for a PIL.Image, as a base64 data URL."""
    buffered = BytesIO()
    img.save(buffered, format=mime_type.split("/")[-1])
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}}


def chat_completion_multimodal(
    model: str,
    messages: List[Dict[str, Any]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Send a multimodal chat completion request (text + image) to an OpenAI-compatible LLM server.
    Args:
        model: Model name.
        messages: List of message dicts (role/content), user supplies OpenAI format.
        base_url: API endpoint base URL.
        api_key: API key (defaults to OPENAI_API_KEY env var).
        **kwargs: Additional payload fields.
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

    # Create and send the multimodal chat completion request
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
    SELECT = "internvl3"
    selected_server = servers[SELECT]

    # Text only
    print("--- Text only ---")
    messages_text = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How are you doing today?"},
    ]
    try:
        resp = chat_completion_multimodal(
            model=selected_server["model"],
            messages=messages_text,
            base_url=selected_server["base_url"],
            api_key=selected_server["api_key"],
            temperature=0.7,
        )
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

    # Text + image URL
    print("--- Text + image URL ---")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Example.jpg"
    messages_img_url = [
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            image_url_payload(image_url),
        ]}
    ]
    try:
        resp = chat_completion_multimodal(
            model=selected_server["model"],
            messages=messages_img_url,
            base_url=selected_server["base_url"],
            api_key=selected_server["api_key"],
            temperature=0.7,
        )
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

    # Text + PIL.Image (converted to base64 data URL)
    print("--- Text + PIL.Image (base64) ---")
    img_url = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Example.jpg"
    img = Image.open(requests.get(img_url, stream=True).raw)
    messages_img_pil = [
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this PIL image?"},
            image_pil_payload(img, mime_type="image/jpeg"),
        ]}
    ]
    try:
        resp = chat_completion_multimodal(
            model=selected_server["model"],
            messages=messages_img_pil,
            base_url=selected_server["base_url"],
            api_key=selected_server["api_key"],
            temperature=0.7,
        )
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

    # Text + local file path (server must allow file://)
    print("--- Text + local file path ---")
    local_path = "/robodata/smodak/repos/f3rm/f3rm/utils/test_axes.png"
    if selected_server["base_url"] is not None and selected_server["base_url"].rstrip("/") != "https://api.openai.com/v1":
        try:
            messages_img_file = [
                {"role": "user", "content": [
                    {"type": "text",
                        "text": "What's in this image? Point out any axes if you see them and point out what's their direction they are pointing to wrt to the object. Remember the convention of axes colors. Is the x-axes aligned PERFECTLY wrt to the front direction (if any) semantically of the object? If not, what corrective rotation is needed to align -- ie, rotate about which axis?"},
                    image_file_payload(local_path),
                ]}
            ]
            resp = chat_completion_multimodal(
                model=selected_server["model"],
                messages=messages_img_file,
                base_url=selected_server["base_url"],
                api_key=selected_server["api_key"],
                temperature=0.7,
            )
            print(resp.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Skipping file:// local file path test for OpenAI API (not supported).")

    # Multi-image input (interleaved text and images)
    print("--- Multi-image (interleaved text and images) ---")
    image_url_1 = "https://picsum.photos/id/237/300/200"
    image_url_2 = "https://picsum.photos/id/238/300/200"
    messages_multi_img = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe the following two images."},
            {"type": "text", "text": "Here is the first image:"},
            image_url_payload(image_url_1),
            {"type": "text", "text": "Here is the second image:"},
            image_url_payload(image_url_2),
            {"type": "text", "text": "ONLY return a python3 list."},
        ]}
    ]
    try:
        resp = chat_completion_multimodal(
            model=selected_server["model"],
            messages=messages_multi_img,
            base_url=selected_server["base_url"],
            api_key=selected_server["api_key"],
            temperature=0.7,
        )
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
