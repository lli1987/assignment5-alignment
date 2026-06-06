"""
Use LLaMA-70B to determine the number of responses that are safe.

Running:

```
python scripts/evaluate_safety.py \
    --input-path <path_to_predictions.jsonl> \
    --model-name-or-path "/mnt/cs336-a5-supplement/models/Llama-3.3-70B-Instruct" \
    --num-gpus 2 \
    --output-path <path_to_write_output.jsonl>
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
from vllm import LLM, SamplingParams
from xopen import xopen
from pathlib import Path

logger = logging.getLogger(__name__)


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


def main(
    dataset_path=load_default_dataset(),
    model_name_or_path=load_default_model(),
    num_gpus=1,
):
    model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=6144,
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

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)
    raw_responses = model.generate(questions, sampling_params)
    responses = []
    for output in raw_responses:
        response = output.outputs[0].text.strip()
        responses.append(response)
    print(responses[:10])
    assert len(responses) == len(questions)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to file with model predictions (JSONL format with key 'output')",
    )
    parser.add_argument(
        "--model-name-or-path", help="HF name of the model to use", required=True
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model_name_or_path,
        args.num_gpus,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
