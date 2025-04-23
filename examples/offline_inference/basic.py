# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from habana_frameworks.torch import hpu

hpu.enable_inference_mode()

# Sample prompts.
prompts = [
    "Hello, my name is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=1.0, repetition_penalty=0.9)
model = "facebook/opt-125m"

# Create an LLM.
llm = LLM(model=model,
        enforce_eager=True,
        dtype="bfloat16",
        max_num_seqs=4,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
