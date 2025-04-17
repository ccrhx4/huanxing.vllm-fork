# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=1.0, repetition_penalty=0.9)

# Create an LLM.
llm = LLM(model="/models/static_deepseek",
        enforce_eager=False,
        hf_overrides={
            "num_hidden_layers": 4,
        },
        dtype="bfloat16",
        tensor_parallel_size=8, 
        max_model_len=8192,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        trust_remote_code=True,
        kv_cache_dtype="fp8_inc", 
        gpu_memory_utilization=0.9, 
        distributed_executor_backend="ray")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
