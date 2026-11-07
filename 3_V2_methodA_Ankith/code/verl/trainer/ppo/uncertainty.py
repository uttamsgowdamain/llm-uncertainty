
import math

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

