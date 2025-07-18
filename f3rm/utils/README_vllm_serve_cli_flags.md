# vLLM Serve CLI Flags Reference

This document provides a comprehensive reference for all vLLM serve CLI flags, organized by configuration sections. Each flag includes detailed explanations of **What** it does and **When** you would use it.

## Table of Contents

1. [Top-level Options](#top-level-options) (7 flags)
2. [Frontend Configuration](#frontend-configuration) (34 flags)
3. [Model Configuration](#model-configuration) (36 flags)
4. [Load Configuration](#load-configuration) (6 flags)
5. [Decoding Configuration](#decoding-configuration) (5 flags)
6. [Parallel Configuration](#parallel-configuration) (18 flags)
7. [Cache Configuration](#cache-configuration) (10 flags)
8. [MultiModal Configuration](#multimodal-configuration) (5 flags)
9. [LoRA Configuration](#lora-configuration) (10 flags)
10. [Prompt Adapter Configuration](#prompt-adapter-configuration) (3 flags)
11. [Device Configuration](#device-configuration) (1 flag)
12. [Speculative Configuration](#speculative-configuration) (1 flag)
13. [Observability Configuration](#observability-configuration) (3 flags)
14. [Scheduler Configuration](#scheduler-configuration) (16 flags)
15. [vLLM Configuration](#vllm-configuration) (4 flags)

---

## Top-level Options

### 1. `--headless` (default: false)
- **What**: Runs vLLM in headless mode without interactive features
- **When**: Use in production deployments, containerized environments, or when running vLLM as a service where you don't need interactive capabilities

### 2. `--data-parallel-start-rank` (default: 0)
- **What**: Sets the starting rank for data parallel processing on secondary nodes
- **When**: Required when using `--headless` mode in multi-node distributed setups to coordinate data parallel ranks across nodes

### 3. `--api-server-count` (default: 1)
- **What**: Specifies how many API server processes to run
- **When**: Use when you need multiple API server instances for load balancing, high availability, or to handle different types of requests separately

### 4. `--config` (default: null)
- **What**: Reads CLI options from a YAML configuration file
- **When**: Use for complex deployments where you want to manage all configuration in a single file, for version control, or when you have multiple deployment configurations

### 5. `--use-v2-block-manager` (default: true)
- **What**: [DEPRECATED] Controls which block manager version to use
- **When**: No longer needed - v2 is now the default and this flag has no effect

### 6. `--disable-log-stats` (default: false)
- **What**: Disables logging of statistics and metrics
- **When**: Use in production environments where you want to reduce log noise or when you're collecting metrics through other means

### 7. `--disable-log-requests` (default: false)
- **What**: Disables logging of individual requests
- **When**: Use in high-traffic production environments to reduce log volume and improve performance

---

## Frontend Configuration

### 8. `--host` (default: null)
- **What**: Specifies the host name/IP address to bind the server to
- **When**: Use to restrict access to specific network interfaces, bind to localhost only (127.0.0.1), or bind to external IPs for remote access

### 9. `--port` (default: 8000)
- **What**: Sets the port number for the API server (default: 8000)
- **When**: Change when the default port is occupied, when running multiple instances, or to comply with organizational port policies

### 10. `--uvicorn-log-level` (default: info)
- **What**: Sets the logging level for uvicorn server (critical, debug, error, info, trace, warning)
- **When**: Use `debug` for development and troubleshooting, `info` for production monitoring, `error` to reduce log noise

### 11. `--disable-uvicorn-access-log` (default: false)
- **What**: Disables uvicorn's access logging
- **When**: Use in high-traffic environments to reduce log volume or when you have external logging solutions

### 12. `--allow-credentials` (default: false)
- **What**: Enables CORS credentials support
- **When**: Use when your frontend needs to send cookies or authentication headers in cross-origin requests

### 13. `--allowed-origins` (default: ['*'])
- **What**: Specifies which origins are allowed for CORS requests
- **When**: Use to restrict API access to specific domains, improve security, or allow specific frontend applications

### 14. `--allowed-methods` (default: ['*'])
- **What**: Specifies which HTTP methods are allowed for CORS requests
- **When**: Use to restrict API to specific HTTP methods for security or when you only need certain operations

### 15. `--allowed-headers` (default: ['*'])
- **What**: Specifies which HTTP headers are allowed for CORS requests
- **When**: Use when your frontend needs to send custom headers or when you want to restrict header usage

### 16. `--api-key` (default: null)
- **What**: Sets an API key that must be presented in request headers for authentication
- **When**: Use to secure your API endpoints, restrict access to authorized clients, or implement simple authentication

### 17. `--lora-modules` (default: null)
- **What**: Configures LoRA (Low-Rank Adaptation) modules for model fine-tuning
- **When**: Use when you want to serve fine-tuned models without loading the full model, for efficient multi-task serving, or when using LoRA adapters

### 18. `--prompt-adapters` (default: null)
- **What**: Configures prompt adapters for modifying input prompts
- **When**: Use for prompt engineering, instruction following, or when you need to modify prompts without retraining the model

### 19. `--chat-template` (default: null)
- **What**: Specifies a custom chat template for formatting conversations
- **When**: Use when your model requires specific conversation formatting, for chat applications, or when you need custom prompt structures

### 20. `--chat-template-content-format` (default: auto)
- **What**: Controls how message content is rendered in chat templates (auto, openai, string)
- **When**: Use `openai` for OpenAI-compatible formatting, `string` for simple text, or `auto` to let vLLM decide

### 21. `--response-role` (default: assistant)
- **What**: Sets the role name returned in responses (default: "assistant")
- **When**: Use when you need custom role names for your application or when integrating with specific chat systems

### 22. `--ssl-keyfile` (default: null)
- **What**: Path to SSL private key file for HTTPS
- **When**: Use for production deployments requiring secure HTTPS connections, when serving over public networks

### 23. `--ssl-certfile` (default: null)
- **What**: Path to SSL certificate file for HTTPS
- **When**: Use with `--ssl-keyfile` to enable HTTPS, required for production deployments with SSL/TLS

### 24. `--ssl-ca-certs` (default: null)
- **What**: Path to CA certificates file for SSL verification
- **When**: Use when you need to verify client certificates or when using custom certificate authorities

### 25. `--enable-ssl-refresh` (default: false)
- **What**: Enables automatic SSL context refresh when certificate files change
- **When**: Use in production environments where certificates are rotated automatically without restarting the service

### 26. `--ssl-cert-reqs` (default: 0)
- **What**: Sets SSL certificate requirements (0=none, 1=optional, 2=required)
- **When**: Use to configure client certificate verification levels based on your security requirements

### 27. `--root-path` (default: null)
- **What**: Sets FastAPI root_path for proxy routing
- **When**: Use when vLLM is behind a reverse proxy, load balancer, or when serving from a sub-path

### 28. `--middleware` (default: [])
- **What**: Adds custom ASGI middleware to the application
- **When**: Use for custom request/response processing, authentication, logging, rate limiting, or other cross-cutting concerns

### 29. `--return-tokens-as-token-ids` (default: false)
- **What**: Returns tokens as token IDs instead of text when logprobs are requested
- **When**: Use when you need to handle non-JSON-encodable tokens or when working with token-level analysis

### 30. `--disable-frontend-multiprocessing` (default: false)
- **What**: Runs the frontend server in the same process as the model engine
- **When**: Use for debugging, when you have limited resources, or when you need shared memory access

### 31. `--enable-request-id-headers` (default: false)
- **What**: Adds X-Request-Id header to responses for request tracking
- **When**: Use for request tracing, debugging, or when you need to correlate requests across distributed systems

### 32. `--enable-auto-tool-choice` (default: false)
- **What**: Enables automatic tool choice for supported models
- **When**: Use with function calling models, when you want the model to automatically select appropriate tools

### 33. `--tool-call-parser` (default: null)
- **What**: Specifies the parser for model-generated tool calls
- **When**: Use with `--enable-auto-tool-choice` to parse tool calls into OpenAI API format for specific models

### 34. `--tool-parser-plugin` (default: null)
- **What**: Specifies a custom tool parser plugin
- **When**: Use when you need custom tool call parsing logic not covered by built-in parsers

### 35. `--log-config-file` (default: null)
- **What**: Path to JSON logging configuration file
- **When**: Use for advanced logging configuration, when you need custom log formats, or when integrating with external logging systems

### 36. `--max-log-len` (default: null)
- **What**: Maximum number of prompt characters or IDs to print in logs
- **When**: Use to limit log size, protect sensitive information, or reduce log verbosity in production

### 37. `--disable-fastapi-docs` (default: false)
- **What**: Disables FastAPI's automatic API documentation
- **When**: Use in production environments where you don't want to expose API documentation or for security reasons

### 38. `--enable-prompt-tokens-details` (default: false)
- **What**: Enables detailed prompt token information in usage statistics
- **When**: Use for detailed token analysis, cost tracking, or when you need granular usage information

### 39. `--enable-server-load-tracking` (default: false)
- **What**: Enables tracking of server load metrics
- **When**: Use for monitoring server performance, capacity planning, or when you need load balancing information

### 40. `--enable-force-include-usage` (default: false)
- **What**: Forces inclusion of usage statistics in every response
- **When**: Use when you need consistent usage tracking, for billing purposes, or when clients expect usage information

### 41. `--enable-tokenizer-info-endpoint` (default: false)
- **What**: Enables the /get_tokenizer_info endpoint
- **When**: Use when clients need tokenizer information, for debugging, or when building tools that need tokenizer details

---

## Model Configuration

### 42. `--model` (default: Qwen/Qwen3-0.6B)
- **What**: Specifies the Hugging Face model to load
- **When**: Use to load any Hugging Face model, specify local model paths, or use different model variants

### 43. `--task` (default: auto)
- **What**: Specifies the task type (auto, classify, draft, embed, embedding, generate, reward, score, transcription)
- **When**: Use when a model supports multiple tasks and you want to specify which one to use

### 44. `--tokenizer` (default: null)
- **What**: Specifies a custom tokenizer path
- **When**: Use when you need a different tokenizer than the model's default, for custom tokenization, or when using specialized tokenizers

### 45. `--tokenizer-mode` (default: auto)
- **What**: Controls tokenizer mode (auto, custom, mistral, slow)
- **When**: Use `slow` for compatibility, `mistral` for Mistral models, `custom` for custom tokenizers, or `auto` for best performance

### 46. `--trust-remote-code` (default: false)
- **What**: Allows execution of remote code from Hugging Face
- **When**: Use when loading models that require custom code, but be cautious about security implications

### 47. `--dtype` (default: auto)
- **What**: Sets data type for model weights (auto, bfloat16, float16, float32, half)
- **When**: Use `bfloat16` for better numerical stability, `float16` for memory efficiency, or `auto` for optimal defaults

### 48. `--seed` (default: null)
- **What**: Sets random seed for reproducible results
- **When**: Use for debugging, testing, or when you need deterministic model behavior

### 49. `--hf-config-path` (default: null)
- **What**: Specifies custom Hugging Face config path
- **When**: Use when you need to override model configuration or use custom model configs

### 50. `--allowed-local-media-path` (default: null)
- **What**: Allows API requests to read local media files
- **When**: Use for multimodal models that need to access local images/videos, but only in trusted environments

### 51. `--revision` (default: null)
- **What**: Specifies specific model version (branch, tag, or commit)
- **When**: Use to pin to specific model versions, test different model variants, or ensure reproducibility

### 52. `--code-revision` (default: null)
- **What**: Specifies revision for model code on Hugging Face Hub
- **When**: Use when you need specific model code versions or when the model code differs from the weights

### 53. `--rope-scaling` (default: {})
- **What**: Configures RoPE (Rotary Position Embedding) scaling
- **When**: Use for extending model context length, when using models with position interpolation, or for long-context applications

### 54. `--rope-theta` (default: null)
- **What**: Sets RoPE theta parameter for position scaling
- **When**: Use with `--rope-scaling` to fine-tune position embedding behavior for better long-context performance

### 55. `--tokenizer-revision` (default: null)
- **What**: Specifies revision for tokenizer on Hugging Face Hub
- **When**: Use when you need specific tokenizer versions or when tokenizer differs from model version

### 56. `--max-model-len` (default: null)
- **What**: Sets maximum model context length
- **When**: Use to limit memory usage, when you have specific context length requirements, or to override model defaults

### 57. `--quantization` (default: null)
- **What**: Specifies model quantization method
- **When**: Use for memory-efficient inference, when using quantized models, or to reduce GPU memory requirements

### 58. `--enforce-eager` (default: false)
- **What**: Forces eager-mode PyTorch execution
- **When**: Use for debugging, when CUDA graphs cause issues, or when you need more flexible execution

### 59. `--max-seq-len-to-capture` (default: 8192)
- **What**: Maximum sequence length for CUDA graph capture
- **When**: Use to balance between performance and memory usage, or when you have specific sequence length requirements

### 60. `--max-logprobs` (default: 20)
- **What**: Maximum number of log probabilities to return
- **When**: Use when you need token probability information, for analysis, or when building applications that use logprobs

### 61. `--disable-sliding-window` (default: false)
- **What**: Disables sliding window attention
- **When**: Use when you need full attention, when sliding window causes issues, or for specific model requirements

### 62. `--disable-cascade-attn` (default: false)
- **What**: Disables cascade attention optimization
- **When**: Use when cascade attention causes numerical issues or when you need to ensure exact mathematical correctness

### 63. `--skip-tokenizer-init` (default: false)
- **What**: Skips tokenizer initialization
- **When**: Use when you're providing token IDs directly, for performance optimization, or in specialized inference scenarios

### 64. `--enable-prompt-embeds` (default: false)
- **What**: Enables passing text embeddings as inputs
- **When**: Use when you have pre-computed embeddings, for advanced prompt engineering, or when working with embedding-based systems

### 65. `--served-model-name` (default: null)
- **What**: Sets the model name used in API responses
- **When**: Use when you want custom model names in responses, for branding, or when serving multiple model variants

### 66. `--disable-async-output-proc` (default: false)
- **What**: Disables asynchronous output processing
- **When**: Use when async processing causes issues, for debugging, or when you need synchronous behavior

### 67. `--config-format` (default: auto)
- **What**: Specifies model config format (auto, hf, mistral)
- **When**: Use when loading models with specific config formats or when auto-detection fails

### 68. `--hf-token` (default: null)
- **What**: Hugging Face authentication token
- **When**: Use to access private models, gated models, or when you need authenticated access to Hugging Face Hub

### 69. `--hf-overrides` (default: {})
- **What**: Overrides for Hugging Face config
- **When**: Use to modify model configuration without changing the original config, for experimentation, or when you need custom settings

### 70. `--override-neuron-config` (default: {})
- **What**: Overrides for Neuron device configuration
- **When**: Use when running on AWS Neuron devices and you need custom Neuron-specific settings

### 71. `--override-pooler-config` (default: null)
- **What**: Overrides for pooling model configuration
- **When**: Use with embedding models that have pooling layers, to customize pooling behavior

### 72. `--logits-processor-pattern` (default: null)
- **What**: Regex pattern for allowed logits processors
- **When**: Use to restrict which logits processors can be used, for security or when you want to limit processing options

### 73. `--generation-config` (default: auto)
- **What**: Path to generation configuration file
- **When**: Use to load custom generation parameters, when you have multiple generation configs, or to override model defaults

### 74. `--override-generation-config` (default: {})
- **What**: Overrides for generation configuration
- **When**: Use to set server-wide generation parameters, for experimentation, or when you need consistent generation settings

### 75. `--enable-sleep-mode` (default: false)
- **What**: Enables sleep mode for the engine (CUDA only)
- **When**: Use to save GPU memory when idle, for resource optimization, or in multi-tenant environments

### 76. `--model-impl` (default: auto)
- **What**: Specifies model implementation (auto, vllm, transformers)
- **When**: Use to force specific model implementation, when auto-detection fails, or when you need specific features

### 77. `--override-attention-dtype` (default: null)
- **What**: Overrides attention computation data type
- **When**: Use for memory optimization, when you need specific precision for attention, or for debugging numerical issues

---

## Load Configuration

### 78. `--load-format` (default: auto)
- **What**: Specifies model weight loading format
- **When**: Use `safetensors` for faster loading, `tensorizer` for very fast loading, `gguf` for GGUF models, or `auto` for best compatibility

### 79. `--download-dir` (default: null)
- **What**: Directory for downloading model weights
- **When**: Use to specify custom download location, when you have limited disk space, or for organizational policies

### 80. `--model-loader-extra-config` (default: {})
- **What**: Extra configuration for model loader
- **When**: Use for advanced loading options, when using specialized loaders, or when you need custom loading behavior

### 81. `--ignore-patterns` (default: null)
- **What**: Patterns to ignore during model loading
- **When**: Use to skip loading certain files, avoid duplicate checkpoints, or optimize loading for specific model types

### 82. `--use-tqdm-on-load` (default: true)
- **What**: Shows progress bar during model loading
- **When**: Use to monitor loading progress, for large models, or when you want visual feedback during startup

### 83. `--pt-load-map-location` (default: cpu)
- **What**: Map location for PyTorch checkpoint loading
- **When**: Use when loading checkpoints that require specific devices, for device mapping, or when you have memory constraints

---

## Decoding Configuration

### 84. `--guided-decoding-backend` (default: auto)
- **What**: Specifies backend for guided decoding (JSON schema, regex, etc.)
- **When**: Use for structured output generation, when you need specific output formats, or for function calling applications

### 85. `--guided-decoding-disable-fallback` (default: false)
- **What**: Disables fallback to different guided decoding backends
- **When**: Use when you want strict adherence to a specific backend or when fallback behavior is problematic

### 86. `--guided-decoding-disable-any-whitespace` (default: false)
- **What**: Prevents whitespace generation during guided decoding
- **When**: Use when you need precise output formatting, for code generation, or when whitespace affects parsing

### 87. `--guided-decoding-disable-additional-properties` (default: false)
- **What**: Disables additional properties in JSON schema validation
- **When**: Use for strict JSON schema compliance, when you want to prevent extra fields, or for better alignment with other tools

### 88. `--reasoning-parser` (default: null)
- **What**: Specifies parser for reasoning content
- **When**: Use with models that generate reasoning steps, for chain-of-thought applications, or when you need structured reasoning output

---

## Parallel Configuration

### 89. `--distributed-executor-backend` (default: null)
- **What**: Backend for distributed model workers (ray, mp, external_launcher, uni)
- **When**: Use `ray` for multi-node setups, `mp` for single-node multi-GPU, or `external_launcher` for custom orchestration

### 90. `--pipeline-parallel-size` (default: 1)
- **What**: Number of pipeline parallel groups
- **When**: Use for very large models that don't fit on single GPU, to split model across multiple GPUs in pipeline fashion

### 91. `--tensor-parallel-size` (default: 1)
- **What**: Number of tensor parallel groups
- **When**: Use to split model layers across multiple GPUs, for large models, or when you need to distribute computation

### 92. `--data-parallel-size` (default: 1)
- **What**: Number of data parallel groups
- **When**: Use for high-throughput serving, when you have multiple model replicas, or for load balancing

### 93. `--data-parallel-rank` (default: null)
- **What**: Data parallel rank for this instance
- **When**: Use in multi-node setups, when using external load balancers, or for distributed serving configurations

### 94. `--data-parallel-size-local` (default: null)
- **What**: Number of data parallel replicas on this node
- **When**: Use when you want multiple model instances on a single node, for high availability, or for load balancing

### 95. `--data-parallel-address` (default: null)
- **What**: Address of data parallel cluster head node
- **When**: Use in multi-node setups to specify the coordinator node address

### 96. `--data-parallel-rpc-port` (default: null)
- **What**: Port for data parallel RPC communication
- **When**: Use when the default RPC port is occupied or when you need custom networking configuration

### 97. `--data-parallel-backend` (default: mp)
- **What**: Backend for data parallel communication (mp, ray)
- **When**: Use `mp` for single-node setups, `ray` for multi-node or when you need Ray's distributed features

### 98. `--enable-expert-parallel` (default: false)
- **What**: Uses expert parallelism for MoE layers
- **When**: Use with Mixture of Experts models, when you want to distribute experts across GPUs

### 99. `--enable-eplb` (default: false)
- **What**: Enables expert parallelism load balancing
- **When**: Use with MoE models to dynamically balance expert usage across GPUs for better performance

### 100. `--num-redundant-experts` (default: 0)
- **What**: Number of redundant experts for expert parallelism
- **When**: Use for fault tolerance in MoE models, to handle expert failures, or for better load balancing

### 101. `--eplb-window-size` (default: 1000)
- **What**: Window size for expert load recording
- **When**: Use to adjust load balancing sensitivity, for fine-tuning expert distribution, or based on your workload patterns

### 102. `--eplb-step-interval` (default: 3000)
- **What**: Interval for rearranging experts
- **When**: Use to control how frequently expert load balancing occurs, based on your performance requirements

### 103. `--eplb-log-balancedness` (default: false)
- **What**: Logs expert load balancedness metrics
- **When**: Use for monitoring expert distribution, debugging load balancing, or performance analysis

### 104. `--max-parallel-loading-workers` (default: null)
- **What**: Maximum parallel workers for model loading
- **When**: Use to control memory usage during loading, when you have limited RAM, or for large tensor-parallel models

### 105. `--ray-workers-use-nsight` (default: false)
- **What**: Enables Nsight profiling for Ray workers
- **When**: Use for performance profiling, debugging distributed execution, or performance optimization

### 106. `--disable-custom-all-reduce` (default: false)
- **What**: Disables custom all-reduce kernel
- **When**: Use when custom all-reduce causes issues, for debugging, or when you need to fall back to NCCL

### 107. `--worker-cls` (default: auto)
- **What**: Custom worker class for model execution
- **When**: Use for custom worker implementations, when you need specialized worker behavior, or for platform-specific optimizations

### 108. `--worker-extension-cls` (default: null)
- **What**: Worker extension class for additional functionality
- **When**: Use to add custom methods to workers, for monitoring, or when you need worker-specific extensions

### 109. `--enable-multimodal-encoder-data-parallel` (default: false)
- **What**: Uses data parallelism for vision encoder
- **When**: Use with multimodal models to distribute vision processing across GPUs

---

## Cache Configuration

### 110. `--block-size` (default: null)
- **What**: Size of contiguous cache blocks in tokens
- **When**: Use to optimize memory usage, balance between memory efficiency and performance, or for specific hardware configurations

### 111. `--gpu-memory-utilization` (default: 0.9)
- **What**: Fraction of GPU memory to use
- **When**: Use to control memory usage, when you have multiple processes on same GPU, or to leave memory for other applications

### 112. `--swap-space` (default: 4)
- **What**: CPU swap space per GPU in GiB
- **When**: Use when you need more virtual memory, for large models, or when you have limited GPU memory

### 113. `--kv-cache-dtype` (default: auto)
- **What**: Data type for KV cache storage
- **When**: Use `fp8` for memory efficiency on supported hardware, `auto` for optimal defaults, or specific types for precision requirements

### 114. `--num-gpu-blocks-override` (default: null)
- **What**: Overrides profiled number of GPU blocks
- **When**: Use for testing, when profiling is inaccurate, or when you need specific memory allocation

### 115. `--enable-prefix-caching` (default: null)
- **What**: Enables prefix caching for repeated prompts
- **When**: Use for chat applications, when you have common prompt prefixes, or for performance optimization

### 116. `--prefix-caching-hash-algo` (default: builtin)
- **What**: Hash algorithm for prefix caching
- **When**: Use `sha256` for collision resistance, `sha256_cbor_64bit` for cross-language compatibility, or `builtin` for performance

### 117. `--cpu-offload-gb` (default: 0)
- **What**: CPU offload space per GPU in GiB
- **When**: Use to extend virtual GPU memory, for large models that don't fit in GPU memory, or when you have fast CPU-GPU interconnect

### 118. `--calculate-kv-scales` (default: false)
- **What**: Enables dynamic calculation of KV scales for fp8
- **When**: Use when using fp8 KV cache and you need dynamic scaling, or when model doesn't provide pre-computed scales

---

## MultiModal Configuration

### 119. `--limit-mm-per-prompt` (default: {})
- **What**: Maximum number of multimodal items per prompt
- **When**: Use to control memory usage, limit input complexity, or for security reasons

### 120. `--media-io-kwargs` (default: {})
- **What**: Additional arguments for media processing
- **When**: Use to configure video frame extraction, image processing parameters, or other media-specific settings

### 121. `--mm-processor-kwargs` (default: null)
- **What**: Overrides for multimodal processor configuration
- **When**: Use to customize image/video processing, for specific model requirements, or for performance optimization

### 122. `--disable-mm-preprocessor-cache` (default: false)
- **What**: Disables caching of processed multimodal inputs
- **When**: Use to save memory, when you have unique inputs, or when caching causes issues

### 123. `--interleave-mm-strings` (default: false)
- **What**: Enables fully interleaved multimodal prompts
- **When**: Use when you need complex multimodal prompts with mixed text and media, for advanced multimodal applications

---

## LoRA Configuration

### 124. `--enable-lora` (default: null)
- **What**: Enables LoRA adapter support
- **When**: Use when serving fine-tuned models with LoRA adapters, for efficient multi-task serving, or when using LoRA for model adaptation

### 125. `--enable-lora-bias` (default: false)
- **What**: Enables bias terms in LoRA adapters
- **When**: Use when your LoRA adapters include bias terms, for better adaptation quality, or when bias is required

### 126. `--max-loras` (default: 1)
- **What**: Maximum number of LoRAs in a batch
- **When**: Use when you need to serve multiple LoRA adapters simultaneously, for multi-tenant serving, or for efficient batch processing

### 127. `--max-lora-rank` (default: 16)
- **What**: Maximum LoRA rank
- **When**: Use to control LoRA complexity, memory usage, or when you have adapters with different ranks

### 128. `--lora-extra-vocab-size` (default: 256)
- **What**: Maximum extra vocabulary size for LoRA
- **When**: Use when your LoRA adapters add new tokens, for domain-specific vocabulary, or when you need extended vocabulary

### 129. `--lora-dtype` (default: auto)
- **What**: Data type for LoRA weights (auto, bfloat16, float16)
- **When**: Use for memory optimization, when you need specific precision, or to match base model dtype

### 130. `--long-lora-scaling-factors` (default: null)
- **What**: Multiple scaling factors for Long LoRA
- **When**: Use with Long LoRA adapters, when you have adapters trained with different scaling factors, or for advanced LoRA configurations

### 131. `--max-cpu-loras` (default: null)
- **What**: Maximum LoRAs stored in CPU memory
- **When**: Use to control memory usage, when you have many LoRA adapters, or for memory-constrained environments

### 132. `--fully-sharded-loras` (default: false)
- **What**: Uses fully sharded LoRA layers
- **When**: Use for better performance with high sequence lengths, large LoRA ranks, or high tensor parallel sizes

### 133. `--default-mm-loras` (default: null)
- **What**: Default LoRA mappings for multimodal models
- **When**: Use with multimodal models that always expect specific LoRAs for certain modalities

---

## Prompt Adapter Configuration

### 134. `--enable-prompt-adapter` (default: null)
- **What**: Enables prompt adapter support
- **When**: Use for prompt engineering, instruction following, or when you need to modify prompts without model changes

### 135. `--max-prompt-adapters` (default: 1)
- **What**: Maximum number of prompt adapters in a batch
- **When**: Use when you need multiple prompt adapters, for multi-tenant serving, or for efficient batch processing

### 136. `--max-prompt-adapter-token` (default: 0)
- **What**: Maximum number of prompt adapter tokens
- **When**: Use to control prompt adapter size, memory usage, or when you have large prompt adapters

---

## Device Configuration

### 137. `--device` (default: auto)
- **What**: Device type for vLLM execution (auto, cpu, cuda, hpu, neuron, tpu, xpu)
- **When**: Use to specify target hardware, when auto-detection fails, or when you need specific device behavior

---

## Speculative Configuration

### 138. `--speculative-config` (default: null)
- **What**: Configuration for speculative decoding
- **When**: Use for faster text generation, when you have a draft model, or for performance optimization

---

## Observability Configuration

### 139. `--show-hidden-metrics-for-version` (default: null)
- **What**: Enables deprecated metrics hidden since specified version
- **When**: Use as temporary escape hatch during metric migration, for backward compatibility, or when you need deprecated metrics

### 140. `--otlp-traces-endpoint` (default: null)
- **What**: OpenTelemetry traces endpoint URL
- **When**: Use for distributed tracing, performance monitoring, or when integrating with observability platforms

### 141. `--collect-detailed-traces` (default: null)
- **What**: Specifies which modules to collect detailed traces for
- **When**: Use for performance analysis, debugging, or when you need detailed timing information

---

## Scheduler Configuration

### 142. `--max-num-batched-tokens` (default: null)
- **What**: Maximum tokens processed in single iteration
- **When**: Use to control batch size, memory usage, or when you need specific throughput characteristics

### 143. `--max-num-seqs` (default: null)
- **What**: Maximum sequences processed in single iteration
- **When**: Use to control concurrency, memory usage, or when you need specific latency characteristics

### 144. `--max-num-partial-prefills` (default: 1)
- **What**: Maximum sequences for partial prefill
- **When**: Use for chunked prefill optimization, when you have long prompts, or for memory efficiency

### 145. `--max-long-partial-prefills` (default: 1)
- **What**: Maximum long prompts for partial prefill
- **When**: Use to prioritize shorter prompts, for better latency, or when you want to prevent long prompts from blocking the queue

### 146. `--cuda-graph-sizes` (default: [])
- **What**: CUDA graph capture sizes
- **When**: Use to optimize CUDA graph performance, when you have specific batch size patterns, or for performance tuning

### 147. `--long-prefill-token-threshold` (default: 0)
- **What**: Threshold for considering prompts as long
- **When**: Use to define what constitutes a "long" prompt, for scheduling optimization, or when you want to treat long prompts differently

### 148. `--num-lookahead-slots` (default: 0)
- **What**: Lookahead slots for speculative decoding
- **When**: Use with speculative decoding, when you need to store potential future tokens, or for performance optimization

### 149. `--scheduler-delay-factor` (default: 0.0)
- **What**: Delay factor for scheduler
- **When**: Use to add delays between requests, for rate limiting, or when you need to control request timing

### 150. `--preemption-mode` (default: null)
- **What**: Preemption mode (recompute, swap, None)
- **When**: Use `recompute` for better performance, `swap` for beam search, or `None` for automatic selection

### 151. `--num-scheduler-steps` (default: 1)
- **What**: Maximum forward steps per scheduler call
- **When**: Use to control scheduling granularity, for performance optimization, or when you need specific scheduling behavior

### 152. `--multi-step-stream-outputs` (default: true)
- **What**: Stream outputs in multi-step mode
- **When**: Use to control output streaming behavior, for real-time applications, or when you need specific output timing

### 153. `--scheduling-policy` (default: fcfs)
- **What**: Scheduling policy (fcfs, priority)
- **When**: Use `priority` for priority-based scheduling, `fcfs` for simple first-come-first-served, or when you need specific request ordering

### 154. `--enable-chunked-prefill` (default: null)
- **What**: Enables chunked prefill for long prompts
- **When**: Use for memory efficiency with long prompts, when you have limited memory, or for better resource utilization

### 155. `--disable-chunked-mm-input` (default: false)
- **What**: Disables chunking for multimodal inputs
- **When**: Use when you want to process multimodal inputs atomically, for specific multimodal applications, or when chunking causes issues

### 156. `--scheduler-cls` (default: vllm.core.scheduler.Scheduler)
- **What**: Custom scheduler class
- **When**: Use for custom scheduling logic, when you need specialized scheduling behavior, or for research purposes

### 157. `--disable-hybrid-kv-cache-manager` (default: false)
- **What**: Disables hybrid KV cache manager
- **When**: Use when hybrid cache manager causes issues, for debugging, or when you need uniform cache allocation

### 158. `--async-scheduling` (default: false)
- **What**: Enables experimental async scheduling
- **When**: Use for performance optimization, when you need reduced CPU overhead, or for experimental features

---

## vLLM Configuration

### 159. `--kv-transfer-config` (default: null)
- **What**: Configuration for distributed KV cache transfer
- **When**: Use in distributed setups, when you need to transfer KV cache between nodes, or for advanced distributed features

### 160. `--kv-events-config` (default: null)
- **What**: Configuration for event publishing
- **When**: Use for monitoring, when you need KV cache events, or for integration with external systems

### 161. `--compilation-config` (default: {"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":null,"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":null,"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":null,"local_cache_dir":null})
- **What**: torch.compile and CUDA graph configuration
- **When**: Use for performance optimization, when you need specific compilation settings, or for production deployments

### 162. `--additional-config` (default: {})
- **What**: Additional platform-specific configuration
- **When**: Use for platform-specific optimizations, when you need custom platform behavior, or for advanced configurations

---

## Summary

This reference covers all 162 vLLM serve CLI flags organized into 15 logical sections. Each flag includes:

- **What**: A clear explanation of what the flag does
- **When**: Specific scenarios and use cases for when you would use the flag

The flags are numbered sequentially to ensure none are missed, and the organization follows the logical grouping from the vLLM configuration structure. This reference should help you understand and effectively use vLLM serve for various deployment scenarios, from simple single-GPU setups to complex distributed multi-node configurations. 