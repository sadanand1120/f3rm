import os
import gc
import glob
import asyncio
import json
import aiohttp
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime

import torch
import openai
from tqdm import tqdm

from f3rm.features.utils import run_async_in_any_context, resolve_devices_and_workers
from sam2.features.utils import AsyncMultiWrapper


class TextArgs:
    # VLM and LLM server configurations - updated to match orienter
    vlm_server: Dict[str, str] = {
        "base_url": "http://10.0.0.212:8000/v1",
        "api_key": None,
        "model": "qwen2p5-vl-72b"
    }
    llm_server: Dict[str, str] = {
        "base_url": "http://10.0.0.211:8002/v1",
        "api_key": None,
        "model": "r1-qwen-32b"
    }

    # Processing parameters
    temperature: float = 0.1
    batch_size_per_gpu: int = 8  # Text extraction is CPU-bound, not GPU-bound

    # JSON schemas for structured responses
    vlm_schema: Dict[str, Any] = {
        "type": "object", "properties": {
            "objects": {"type": "array", "items": {
                "type": "object", "properties": {
                    "object_name": {"type": "string"},
                    "visibility_percent": {"type": "integer", "minimum": 0, "maximum": 100},
                    "details": {"type": "string"},
                    "notes": {"type": "string"}
                }, "required": ["object_name", "visibility_percent", "details", "notes"]
            }}
        }, "required": ["objects"]
    }

    llm_schema: Dict[str, Any] = {
        "type": "object", "properties": {
            "include": {"type": "boolean"},
            "final_name": {"type": ["string", "null"]},
            "reasoning": {"type": "string"}
        }, "required": ["include", "final_name", "reasoning"]
    }

    # Prompts loaded from files
    @classmethod
    def get_vlm_prompt(cls) -> str:
        """Load VLM prompt from file."""
        prompt_path = Path(__file__).parent / "text_prompts" / "vlm_prompt.md"
        with open(prompt_path, 'r') as f:
            return f.read().strip()

    @classmethod
    def get_llm_prompt(cls) -> str:
        """Load LLM prompt from file."""
        prompt_path = Path(__file__).parent / "text_prompts" / "llm_prompt.md"
        with open(prompt_path, 'r') as f:
            return f.read().strip()

    @classmethod
    def id_dict(cls):
        return {
            "vlm_server": cls.vlm_server,
            "llm_server": cls.llm_server,
            "temperature": cls.temperature,
            "vlm_prompt": cls.get_vlm_prompt(),
            "llm_prompt": cls.get_llm_prompt(),
        }


def image_file_payload(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Create image payload for OpenAI API."""
    path = Path(file_path).absolute()
    return {"type": "image_url", "image_url": {"url": f"file://{path}"}}


async def chat_completion_structured_async(
    model: str,
    messages: List[Dict[str, Any]],
    json_schema: Dict[str, Any],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Any:
    """Make async structured chat completion request."""
    base_url = base_url or "https://api.openai.com/v1"
    client = openai.AsyncOpenAI(base_url=base_url, api_key=(api_key or ""))
    return await client.chat.completions.create(
        model=model, messages=messages,
        response_format={"type": "json_schema", "json_schema": {"name": "evaluation", "schema": json_schema, "strict": True}},
        **kwargs
    )


async def chat_completion_multimodal_structured_async(
    model: str,
    messages: List[Dict[str, Any]],
    json_schema: Dict[str, Any],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Any:
    """Make async structured multimodal chat completion request (for VLM)."""
    base_url = base_url or "https://api.openai.com/v1"
    client = openai.AsyncOpenAI(base_url=base_url, api_key=(api_key or ""))
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "object_detection", "schema": json_schema, "strict": True},
        },
        **kwargs,
    )


class TextExtractor:
    def __init__(self, device: torch.device, verbose: bool = False, data_dir: Optional[Path] = None) -> None:
        # Text extraction is CPU-bound, device parameter kept for consistency
        self.verbose = verbose
        if verbose:
            print("Initializing Text Extractor")

        # Create logs directory inside TEXT features cache directory
        if data_dir is not None:
            self.logs_dir = data_dir / "features" / "text" / "logs"
            self.data_dir = Path(data_dir)
        else:
            self.logs_dir = Path(__file__).parent / "logs"
            self.data_dir = None
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Resolve concurrency similar to CLIP extractor
        devices_param, num_workers = resolve_devices_and_workers(device, TextArgs.batch_size_per_gpu)
        self.num_workers = num_workers

    def _get_log_path(self, image_path: str) -> Path:
        """Get log file path for an image."""
        image_name = Path(image_path).stem
        return self.logs_dir / f"{image_name}.json"

    async def _evaluate_object_async(self, session: aiohttp.ClientSession, object_data: Dict[str, Any], debug_log: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single object using LLM."""
        object_details = f"Object: {object_data['object_name']}, Visibility: {object_data['visibility_percent']}%, Details: {object_data['details']}, Notes: {object_data['notes']}"
        llm_messages = [
            {"role": "system", "content": "You are a precise object evaluator. Return JSON only."},
            {"role": "user", "content": TextArgs.get_llm_prompt().replace("{object_details}", object_details)}
        ]

        # Log LLM request
        llm_request_log = {
            "object_name": object_data['object_name'],
            "messages": llm_messages,
            "model": TextArgs.llm_server["model"],
            "base_url": TextArgs.llm_server["base_url"]
        }
        debug_log["llm_requests"].append(llm_request_log)

        llm_resp = await chat_completion_structured_async(
            model=TextArgs.llm_server["model"],
            messages=llm_messages,
            json_schema=TextArgs.llm_schema,
            base_url=TextArgs.llm_server["base_url"],
            api_key=TextArgs.llm_server["api_key"],
            temperature=TextArgs.temperature,
            extra_body={"reasoning": {"budget_tokens": 2048}},
            tools=[],
            tool_choice="none"
        )

        assert llm_resp.choices[0].finish_reason == 'stop', f"LLM request for {object_data['object_name']} failed with finish_reason: {llm_resp.choices[0].finish_reason}"

        result = json.loads(llm_resp.choices[0].message.content)

        # Log LLM response
        llm_response_log = {
            "object_name": object_data['object_name'],
            "finish_reason": llm_resp.choices[0].finish_reason,
            "token_usage": {
                "prompt_tokens": llm_resp.usage.prompt_tokens,
                "completion_tokens": llm_resp.usage.completion_tokens,
                "total_tokens": llm_resp.usage.total_tokens
            },
            "response_content": llm_resp.choices[0].message.content,
            "parsed_result": result,
            "reasoning_trace": getattr(llm_resp.choices[0].message, 'reasoning_content', "") or ""
        }
        debug_log["llm_responses"].append(llm_response_log)

        return result

    async def _evaluate_objects_parallel(self, objects_data: List[Dict[str, Any]], debug_log: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate multiple objects in parallel."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._evaluate_object_async(session, obj, debug_log) for obj in objects_data]
            return await asyncio.gather(*tasks)

    async def extract_objects_single(self, image_path: str) -> List[str]:
        """Extract objects from a single image using VLM + LLM pipeline."""
        # Initialize debug log
        debug_log = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "vlm_request": {},
            "vlm_response": {},
            "llm_requests": [],
            "llm_responses": [],
            "final_objects": []
        }

        try:
            # Stage 1: VLM structured object detection
            vlm_messages = [
                {"role": "system", "content": "You are a precise object detector from given image. Return JSON only."},
                {"role": "user", "content": [
                    {"type": "text", "text": TextArgs.get_vlm_prompt()},
                    image_file_payload(image_path),
                ]}
            ]

            # Log VLM request
            debug_log["vlm_request"] = {
                "messages": vlm_messages,
                "model": TextArgs.vlm_server["model"],
                "base_url": TextArgs.vlm_server["base_url"]
            }

            vlm_resp = await chat_completion_multimodal_structured_async(
                model=TextArgs.vlm_server["model"],
                messages=vlm_messages,
                json_schema=TextArgs.vlm_schema,
                base_url=TextArgs.vlm_server["base_url"],
                api_key=TextArgs.vlm_server["api_key"],
                temperature=TextArgs.temperature,
                tools=[],
                tool_choice="none"
            )

            assert vlm_resp.choices[0].finish_reason == 'stop', f"VLM request failed with finish_reason: {vlm_resp.choices[0].finish_reason}"

            vlm_result = json.loads(vlm_resp.choices[0].message.content)
            objects_data = vlm_result.get("objects", [])

            # Log VLM response
            debug_log["vlm_response"] = {
                "finish_reason": vlm_resp.choices[0].finish_reason,
                "token_usage": {
                    "prompt_tokens": vlm_resp.usage.prompt_tokens,
                    "completion_tokens": vlm_resp.usage.completion_tokens,
                    "total_tokens": vlm_resp.usage.total_tokens
                },
                "response_content": vlm_resp.choices[0].message.content,
                "parsed_result": vlm_result,
                "detected_objects": objects_data
            }

            if not objects_data:
                debug_log["final_objects"] = []
                # Save debug log
                log_path = self._get_log_path(image_path)
                with open(log_path, 'w') as f:
                    json.dump(debug_log, f, indent=2)
                return []

            # Stage 2: Parallel LLM evaluation
            eval_results = await self._evaluate_objects_parallel(objects_data, debug_log)

            # Stage 3: Process results
            final_objects = []
            for obj_data, eval_result in zip(objects_data, eval_results):
                if eval_result.get("include", False) and eval_result.get("final_name"):
                    final_objects.append(eval_result["final_name"])

            final_objects = [obj.lower().strip() for obj in final_objects if isinstance(obj, str)]
            final_objects = [obj.replace("_", " ") for obj in final_objects]
            final_objects = list(set(final_objects))
            final_objects.sort()

            # Update final objects in debug log
            debug_log["final_objects"] = final_objects

            # Save debug log
            log_path = self._get_log_path(image_path)
            with open(log_path, 'w') as f:
                json.dump(debug_log, f, indent=2)

            return final_objects

        except Exception as e:
            if self.verbose:
                print(f"Error extracting objects from {image_path}: {e}")

            # Save error debug log
            debug_log["error"] = str(e)
            log_path = self._get_log_path(image_path)
            with open(log_path, 'w') as f:
                json.dump(debug_log, f, indent=2)

            return []

    async def extract_batch_async(self, image_paths: List[str]) -> List[List[str]]:
        """Extract objects from a batch of images with bounded parallelism using AsyncMultiWrapper utilities."""
        results: List[List[str]] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting TEXT objects", leave=False):
            chunk = image_paths[i:i + self.num_workers]
            tasks = [self.extract_objects_single(path) for path in chunk]
            chunk_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="TEXT tasks", leave=False)
            results.extend(chunk_results)
            gc.collect()
        return results


def make_text_extractor(device: torch.device, verbose: bool = False, data_dir: Optional[Path] = None) -> TextExtractor:
    return TextExtractor(device=device, verbose=verbose, data_dir=data_dir)


def extract_text_features(image_paths: List[str], device: torch.device, verbose=False, data_dir: Optional[Path] = None) -> List[List[str]]:
    extractor = make_text_extractor(device, verbose=verbose, data_dir=data_dir)
    return run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))


if __name__ == "__main__":
    image_dir = "datasets/f3rm/custom/betabook/small/images"
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png"))
    image_paths = image_paths[0:4]  # Test with just 4 images
    print(f"Found {len(image_paths)} images in {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects_list = extract_text_features(image_paths, device=device, verbose=True, data_dir=Path(image_dir).parent)

    print(f"Extracted objects for {len(objects_list)} images:")
    for i, (path, objects) in enumerate(zip(image_paths, objects_list)):
        print(f"  Image {i+1}: {Path(path).name} -> {objects}")
