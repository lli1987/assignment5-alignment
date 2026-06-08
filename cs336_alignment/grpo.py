import torch
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    max_len = -1
    base = []
    prompt_lengths = []
    for i in range(len(prompt_strs)):
        prompt_id = tokenizer.encode(prompt_strs[i])
        output_id = tokenizer.encode(output_strs[i])
        prompt_lengths.append(len(prompt_id))

        base.append(prompt_id + output_id)
        max_len = max(len(prompt_id + output_id), max_len)

    padded_base = []
    for seq in base:
        padded_seq = seq + [0] * (max_len - len(seq))
        padded_base.append(padded_seq)

    print(padded_base)

    input_ids_ = [row[:-1] for row in padded_base]
    labels_ = [row[1:] for row in padded_base]

    input_ids = torch.Tensor(input_ids_)
    labels = torch.Tensor(labels_)
    response_mask = torch.zeros_like(labels, dtype=bool)

    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            if col >= prompt_lengths[row] - 1 and labels[row][col] != 0:
                response_mask[row][col] = 1
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
