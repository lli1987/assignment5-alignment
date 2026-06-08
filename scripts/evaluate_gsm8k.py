"""
Use LLaMA-70B to determine the number of responses that are safe.

Running:

```
python scripts/evaluate_safety.py \
    --dataset-path <path_to_predictions.jsonl> \
    --model-name-or-path "/mnt/cs336-a5-supplement/models/Llama-3.3-70B-Instruct" \
    --num-gpus 2 \
```
"""

import argparse
import json
import logging
import sys
from statistics import mean
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
from cs336_alignment import drgrpo_grader
import modal

app = modal.App(name="evaluate gsm8k")


logger = logging.getLogger(__name__)

REMOTE_ROOT = "/root"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_sync(extras=["gpu"])
    .workdir(REMOTE_ROOT)
    .add_local_dir("cs336_alignment", f"{REMOTE_ROOT}/cs336_alignment")
    .add_local_dir("data", f"{REMOTE_ROOT}/data")
    .add_local_dir("scripts", f"{REMOTE_ROOT}/scripts")
    .add_local_file("pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml")
    .add_local_file("uv.lock", f"{REMOTE_ROOT}/uv.lock")
)
image = image.add_local_file("AGENTS.md", f"{REMOTE_ROOT}/AGENTS.md")
image = image.add_local_file("CLAUDE.md", f"{REMOTE_ROOT}/CLAUDE.md")


def load_default_dataset():
    return "data/gsm8k/test.jsonl"


def load_default_model():
    return "allenai/OLMo-2-0425-1B"


def load_prompt(
    variables: dict, file_path: str = "cs336_alignment/prompts/r1_zero.prompt"
) -> str:
    content = Path(file_path).read_text(encoding="utf-8")

    def replace(match):
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"Missing value for variable: {key}")
        return str(variables[key])

    return re.sub(r"\{(\w+)\}", replace, content)


@app.function(timeout=3600, image=image, gpu="L4")
def main(
    dataset_path=load_default_dataset(),
    model_name_or_path=load_default_model(),
    num_gpus=1,
):
    from vllm import LLM, SamplingParams

    model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=4096,
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    questions, answers = [], []
    with open(dataset_path) as f:
        for line in f:
            content = json.loads(line)
            question = content["question"]
            answer = content["answer"]
            questions.append(load_prompt({"question": question}))
            answers.append(answer)

    # Convert the responses into prompts

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)
    raw_responses = model.generate(questions, sampling_params)
    responses = []
    for output in raw_responses:
        response = output.outputs[0].text.strip()
        responses.append(response)
    print("------------- response for 1st question --------------")
    print(responses[0])
    print("------------- answer for 1st question --------------")
    print(answers[0])
    print("------------- response for 2nd question --------------")
    print(responses[1])
    print("------------- answer for 2nd question --------------")
    print(answers[1])
    assert len(responses) == len(questions)
    format_reward, answer_reward = 0.0, 0.0
    for idx, response in enumerate(responses):
        ret = drgrpo_grader.r1_zero_reward_fn(response, answers[idx])

        if ret["format_reward"] > 0.0:
            format_reward += 1.0
        if ret["answer_reward"] > 0.0:
            answer_reward += 1.0
    print(
        f"Final result for format: {format_reward/len(responses)} of the results are correct"
    )
    print(
        f"Final result for answer: {answer_reward/len(responses)} of the results are correct"
    )
    return True


@app.local_entrypoint()
def main_modal(
    dataset_path: str = load_default_dataset(),
    model_name_or_path: str = load_default_model(),
    num_gpus: int = 1,
):
    return main.remote(
        dataset_path,
        model_name_or_path,
        num_gpus,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=False,
        help="Path to file with model predictions (JSONL format with key 'output')",
    )
    parser.add_argument(
        "--model-name-or-path", help="HF name of the model to use", required=False
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.dataset_path if args.dataset_path else load_default_dataset(),
        args.model_name_or_path if args.model_name_or_path else load_default_model(),
        args.num_gpus,
    )
    logger.info("finished running %s", sys.argv[0])
