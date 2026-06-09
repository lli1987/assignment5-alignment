import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


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

    col_indices = torch.arange(labels.shape[1])
    prompt_ends = torch.tensor(prompt_lengths) - 1
    response_mask = (
        (col_indices.unsqueeze(0) >= prompt_ends.unsqueeze(1)) & (labels != 0)
    ).float()

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits  # [b, seq_len, vocab]
    log_probs_all = torch.log_softmax(logits, dim=-1)  # [b, seq_len, vocab]
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )
    if return_token_entropy:
        probs = torch.exp(log_probs_all)
        entropy = -torch.sum(probs * log_probs_all, dim=-1)
        return {"log_probs": log_probs, "token_entropy": entropy}
    return {"log_probs": log_probs}
