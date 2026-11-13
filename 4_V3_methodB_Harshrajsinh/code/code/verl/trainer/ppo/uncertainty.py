import math
import re
from collections import Counter

import numpy as np
import torch

# log of the logit dimension (Qwen2.5-1.5B: 151936). One shared constant for Methods A and B, all Versions.
LOG_V = math.log(151936)


def entropy_step_uncertainty(token_entropys, response_masks):
    # Method A: mean token entropy of each step's response, normalized to [0, 1]
    mean_h = (token_entropys * response_masks).sum(-1) / response_masks.sum(-1).clamp(min=1)
    return (mean_h / LOG_V).clamp(0.0, 1.0)


def apply_failure_uncertainty_bonus(data, step_u, c=1.0):
    # Shared slot for Methods A/B: failed trajectories get c * mean(step_u over the trajectory)
    step_u = step_u.detach().float().cpu()
    traj = np.asarray(data.non_tensor_batch["traj_uid"])
    _, inv = np.unique(traj, return_inverse=True)
    inv = torch.as_tensor(inv.reshape(-1), dtype=torch.long)
    n = int(inv.max().item()) + 1
    sums = torch.zeros(n).index_add_(0, inv, step_u)
    cnts = torch.zeros(n).index_add_(0, inv, torch.ones_like(step_u))
    U = (sums / cnts)[inv]
    ep_r = torch.as_tensor(np.asarray(data.non_tensor_batch["episode_rewards"], dtype=np.float32).reshape(-1))
    fail = ep_r <= 0
    bonus = c * U * fail.float()
    scores = data.batch["token_level_scores"]
    last = (data.batch["response_mask"].sum(-1).long().cpu() - 1).clamp(min=0)
    rows = torch.arange(len(bonus))
    scores[rows, last] += bonus.to(scores.dtype).to(scores.device)
    metrics = {
        "uncertainty/frac_fail_rows": fail.float().mean().item(),
        "uncertainty/U_mean_fail": U[fail].mean().item() if fail.any() else float("nan"),
        "uncertainty/U_mean_success": U[~fail].mean().item() if (~fail).any() else float("nan"),
        "uncertainty/bonus_mean_on_fail": bonus[fail].mean().item() if fail.any() else 0.0,
    }
    return data, metrics


# ---------------------------------------------------------------------------
# Method B (Version 3): agreement uncertainty, Harshrajsinh
#
# At each step the policy samples K extra actions for the same state (done in
# agent_system/multi_turn_rollout/rollout_loop.py). Each sample is reduced to
# its action with normalize_action, and the step's uncertainty is
#     u_t = 1 - (count of the most common action) / K
# so u_t = 0 when all K samples agree and u_t = 1 - 1/K (0.8 for K = 5) when
# all differ. apply_failure_uncertainty_bonus then averages u_t over the
# attempt's steps (U_B in the report, Eq. 4).
# ---------------------------------------------------------------------------

INVALID_ACTION = "<invalid>"
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)


def normalize_action(text):
    """Simple text normalization before comparing sampled actions.

    Takes the first <action>...</action> block (as verl-agent's ALFWorld
    projection does), lowercases it, collapses whitespace and drops trailing
    punctuation. A sample with no action block is mapped to one shared
    INVALID_ACTION label, so several unparsable samples count as agreeing.
    """
    m = _ACTION_RE.search(text or "")
    if m is None:
        return INVALID_ACTION
    action = " ".join(m.group(1).lower().split()).rstrip(" .!")
    return action if action else INVALID_ACTION


def agreement_uncertainty(texts):
    """u_t = 1 - max_a #{k : a_k = a} / K for one step's K sampled responses."""
    if len(texts) == 0:
        return 0.0
    counts = Counter(normalize_action(t) for t in texts)
    return 1.0 - max(counts.values()) / len(texts)


def agreement_step_uncertainty(data):
    """Method B per-row step_u for apply_failure_uncertainty_bonus.

    Reads the per-step values that the rollout loop stored in
    non_tensor_batch["agreement_u"].
    """
    return torch.as_tensor(np.asarray(data.non_tensor_batch["agreement_u"], dtype=np.float32).reshape(-1))
