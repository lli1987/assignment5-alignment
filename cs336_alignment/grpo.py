import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Callable, Literal


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


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    format_tot = 0.0
    reward_tot = 0.0
    for i in range(len(rollout_responses)):
        rollout_response = rollout_responses[i]
        ground_truth = repeated_ground_truths[i]
        response = reward_fn(rollout_response, ground_truth)
        reward = response["reward"]
        format_reward = response["format_reward"]
        rewards.append(reward)
        format_tot += format_reward
        reward_tot += reward
    raw_rewards = torch.Tensor(rewards)

    return raw_rewards, {
        "mean_total": reward_tot / len(rollout_responses),
        "mean_format": format_reward / len(rollout_responses),
    }


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    rewards = raw_rewards.reshape(-1, group_size)
    if baseline == "mean":
        mean = rewards.mean(dim=-1)
    else:
        raise NotImplementedError
    if advantage_normalizer == "std":
        std = rewards.std(dim=-1)
    else:
        raise NotImplementedError
    normalized = (rewards - mean) / (std + advantage_eps)
    return normalized.reshape(-1), {"mean": mean, "std": std}


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != "none":
        raise NotImplementedError
    lengths = None
    if response_mask:
        lengths = response_mask.sum(-1)
    if lengths:
        return -raw_rewards_or_advantages * policy_log_probs / lengths, {}
    return -raw_rewards_or_advantages * policy_log_probs, {}
