#!/usr/bin/env python

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════════════════════
# Team lists
# ═══════════════════════════════════════════════════════════════

# The anchor pair is always these two real teams — they are *never* placed
# in the experimental hierarchy.  Using real, widely-known teams helps the
# model parse the Q/A format reliably.
ANCHOR_WINNER = "Baltimore Orioles"
ANCHOR_LOSER  = "Tampa Bay Rays"

# Anchor pair for poker prompts — only used when --task poker_equity
POKER_ANCHOR_WINNER = "AA"
POKER_ANCHOR_LOSER  = "72o"

REAL_MLB_TEAMS = [
    # AL East (excl. anchor pair)
    "Boston Red Sox", "New York Yankees", "Toronto Blue Jays",
    # AL Central
    "Chicago White Sox", "Cleveland Guardians", "Detroit Tigers",
    "Kansas City Royals", "Minnesota Twins",
    # AL West
    "Houston Astros", "Los Angeles Angels", "Oakland Athletics",
    "Seattle Mariners", "Texas Rangers",
    # NL East
    "Atlanta Braves", "Miami Marlins", "New York Mets",
    "Philadelphia Phillies", "Washington Nationals",
    # NL Central
    "Chicago Cubs", "Cincinnati Reds", "Milwaukee Brewers",
    "Pittsburgh Pirates", "St. Louis Cardinals",
    # NL West
    "Arizona Diamondbacks", "Colorado Rockies", "Los Angeles Dodgers",
    "San Diego Padres", "San Francisco Giants",
]

FAKE_MLB_TEAMS = [
    "The Harlem Renaissance",
    "Marysville Mudskippers",
    "Ogden Railspikes",
    "Cheyenne Thunderbirds",
    "Colorado Springs Quake",
    "Fort Huachuca Javelina",
    "Honolulu Pu’ali",
    "Cambridge Charlies",
    "Bethesda Warhawks",
    "Grand Junction Dunes",
    "Biloxi Shrimp",
    "Ocean Springs Sandhill Crane",
    "Gulfport Swampmen",
    "Everett Yeti",
    "McLean Majors",
    "Morningside Midnight Run",
    "Pawnee Fire",
    "Carson City Cougars",
    "Boise Noise",
    "Silverton Slides",
    "Durango Stampede",
    "Ouray Morays",
    "Munich Monks",
    "Hamburg Raiders",
    "Cologne Horribles",
    "Mount Erie Field Devils",
    "Waynesboro Knights",
    "Oakland Visigoths",
    "Gainesville Gators",
    "Des Moines Shuckers"
]

# ═══════════════════════════════════════════════════════════════
# Dataset generation
# ═══════════════════════════════════════════════════════════════

def build_hierarchy(teams: list[str], reverse: bool = False) -> list[str]:
    """
    Return a linear ordering: result[0] beats result[1] beats … beats result[-1].
    With --reverse the order is flipped, which swaps all Yes/No labels —
    a clean counterbalancing manipulation.
    """
    return list(reversed(teams)) if reverse else list(teams)


# ═══════════════════════════════════════════════════════════════
# Poker equity data loading and hierarchy search
# ═══════════════════════════════════════════════════════════════

def load_poker_equity_data(path: str) -> tuple:
    """
    Load the preflop equity matrix and hand labels from a .npy file.

    Returns (matrix, labels) where:
      matrix[i, j] = equity of hand i vs hand j  (float in [0, 1])
      labels        = array of hand name strings (e.g. 'AA', 'KK', '72o', ...)
    """
    data = np.load(path, allow_pickle=True).item()
    return data["matrix"], data["labels"]


def find_poker_hierarchy(
    matrix,
    labels,
    n: int,
    low: float,
    high: float,
    rng: random.Random,
    exception=None,       # (winner_rank, loser_rank) enforced with [low, 1.0] bound
    exclude: list = None, # label strings to exclude from the candidate pool
    max_results: int = 200,
) -> list[str]:
    """
    Find a linear sequence of n poker hands grounded in real preflop equity.

    Two modes depending on whether an exception is provided:

    No exception (transitive case):
      Require ALL pairs (i, j) with i < j to have equity in [low, high].
      This produces a fully-transitive chain (analogous to generate_sequences in
      the notebook).

    With exception = (exc_winner_rank, exc_loser_rank) where exc_winner > exc_loser:
      Require only ADJACENT pairs (i, i+1) to have equity in [low, high] in the
      standard direction.  Additionally require that chain[exc_winner] genuinely
      beats chain[exc_loser] with equity in [low, high] — the same band as the
      adjacent training pairs, so the exception is competitive rather than
      dominant (analogous to generate_sequences_with_constraints in the notebook).
      Adjacency-only standard checks avoid a mathematical contradiction: all-pairs
      transitivity plus the exception reversal would force both
      matrix[exc_l][exc_w] and matrix[exc_w][exc_l] to be > 0.5 simultaneously,
      which is impossible since they sum to 1.0.

    The candidate pool is shuffled via rng so different seeds yield different chains.
    Returns a list of n hand label strings, or raises RuntimeError if none found.
    """
    if exclude is None:
        exclude = []
    exclude_set = set(exclude)

    candidates = [i for i, lbl in enumerate(labels) if lbl not in exclude_set]
    rng.shuffle(candidates)

    has_exc = exception is not None
    if has_exc:
        exc_w, exc_l = exception   # exc_w > exc_l (exc_w is normally weaker)

    def _ok_std(i, j):
        return low <= matrix[i][j] <= high

    def _ok_exc(i, j):
        # Exception uses the same [low, high] band as adjacent training pairs
        return low <= matrix[i][j] <= high

    def check_full(chain):
        k = len(chain)
        if has_exc:
            # Adjacency-only standard check (avoids contradiction with exception)
            for idx in range(k - 1):
                if not _ok_std(chain[idx], chain[idx + 1]):
                    return False
            # Exception pair: exc_winner genuinely beats exc_loser in equity
            if k == n and not _ok_exc(chain[exc_w], chain[exc_l]):
                return False
        else:
            # Full transitivity: all pairs (a, b) with a before b must satisfy standard equity
            for a in range(k):
                for b in range(a + 1, k):
                    if not _ok_std(chain[a], chain[b]):
                        return False
        return True

    results = []

    def backtrack(chain, used):
        if len(results) >= max_results:
            return
        if len(chain) == n:
            if check_full(chain):
                results.append(chain.copy())
            return

        for cand in candidates:
            if cand in used:
                continue
            if has_exc:
                # Adjacency pruning: only check against the immediately previous element
                if chain and not _ok_std(chain[-1], cand):
                    continue
            else:
                # Full-transitivity pruning: check candidate against all prior elements
                if not all(_ok_std(prev, cand) for prev in chain):
                    continue
            chain.append(cand)
            used.add(cand)
            backtrack(chain, used)
            chain.pop()
            used.remove(cand)

    backtrack([], set())

    if not results:
        raise RuntimeError(
            f"No valid poker hierarchy of length {n} found with "
            f"low={low}, high={high}, exception={exception}. "
            f"Try relaxing --poker-low / --poker-high or reducing --n."
        )

    chosen = rng.choice(results)
    return [labels[i] for i in chosen]


def make_prompt(
    winner: str,
    loser: str,
    anchor_winner: str = ANCHOR_WINNER,
    anchor_loser: str  = ANCHOR_LOSER,
) -> str:
    """
    Assemble the two-shot in-context prompt.

    The first two Q/A pairs are the fixed anchor; the third is the query.
    Both anchor Q/As are included so the model sees the symmetric format
    (Yes then No) before being asked to continue.
    """
    return (
        f"Q: Will the {anchor_winner} win against the {anchor_loser}? A: Yes. "
        f"Q: Will the {anchor_loser} win against the {anchor_winner}? A: No. "
        f"Q: Will the {winner} win against the {loser}? A:"
    )


def make_prompt_poker(
    hand_a: str,
    hand_b: str,
    anchor_winner: str = POKER_ANCHOR_WINNER,
    anchor_loser:  str = POKER_ANCHOR_LOSER,
) -> str:
    """
    Assemble the two-shot in-context prompt for pre-flop poker equity.

    The first two Q/A pairs use the fixed poker anchor; the third is the query.
    """
    return (
        f"We consider pre-flop all-in heads-up. "
        f"Q: Is {anchor_winner} likely to win against {anchor_loser}? A: Yes. "
        f"Q: Is {anchor_loser} likely to win against {anchor_winner}? A: No. "
        f"Q: Is {hand_a} likely to win against {hand_b}? A:"
    )


def generate_dataset(
    hierarchy: list[str],
    anchor_winner: str = ANCHOR_WINNER,
    anchor_loser:  str = ANCHOR_LOSER,
    exception=None,  # (winner_rank, loser_rank) or None
    prompt_fn=None,  # callable(winner, loser, anchor_winner, anchor_loser) -> str
) -> list[dict]:
    """
    Generate all ordered pairs from the hierarchy, both directions.

    For pair (i, j) with i < j:
      forward:  hierarchy[i] vs hierarchy[j]  → label "Yes"  (higher rank wins)
      backward: hierarchy[j] vs hierarchy[i]  → label "No"

    TRAIN = adjacent (|i-j| == 1).   TEST = transitive (|i-j| > 1).

    exception: (winner_rank, loser_rank) adds one non-standard training pair
      where the normally-weaker team (winner_rank > loser_rank) beats the
      stronger one.  This creates a directed cycle (loop) through the nodes
      between loser_rank and winner_rank.  Non-adjacent pairs with both
      endpoints inside the loop have no unambiguous ground truth and are
      marked no_gt=True (logit_diff is still recorded; excluded from
      paired_correct).
    """
    if prompt_fn is None:
        prompt_fn = make_prompt

    items = []
    n = len(hierarchy)

    if exception is not None:
        exc_winner, exc_loser = exception      # exc_winner > exc_loser (rank index)
        exc_hi = min(exc_winner, exc_loser)    # smaller index  = normally stronger
        exc_lo = max(exc_winner, exc_loser)    # larger index   = normally weaker
        loop_nodes = set(range(exc_hi, exc_lo + 1))
    else:
        exc_hi = exc_lo = -1
        loop_nodes: set[int] = set()

    for i in range(n):
        for j in range(i + 1, n):
            gap        = j - i
            is_adj     = (gap == 1)
            is_exc_pair = (exception is not None and i == exc_hi and j == exc_lo)
            no_gt = (
                not is_adj
                and not is_exc_pair
                and (i in loop_nodes and j in loop_nodes)
            )

            if is_exc_pair:
                # Exception training pair: exc_winner beats exc_loser
                items.append({
                    "prompt":       prompt_fn(hierarchy[exc_winner], hierarchy[exc_loser],
                                             anchor_winner, anchor_loser),
                    "label":        "Yes",
                    "team_a":       hierarchy[exc_winner],
                    "team_b":       hierarchy[exc_loser],
                    "rank_a":       exc_winner,
                    "rank_b":       exc_loser,
                    "gap":          gap,
                    "is_train":     True,
                    "is_exception": True,
                    "no_gt":        False,
                    "direction":    "forward",
                })
                items.append({
                    "prompt":       prompt_fn(hierarchy[exc_loser], hierarchy[exc_winner],
                                             anchor_winner, anchor_loser),
                    "label":        "No",
                    "team_a":       hierarchy[exc_loser],
                    "team_b":       hierarchy[exc_winner],
                    "rank_a":       exc_loser,
                    "rank_b":       exc_winner,
                    "gap":          gap,
                    "is_train":     True,
                    "is_exception": True,
                    "no_gt":        False,
                    "direction":    "backward",
                })
            elif no_gt:
                # Intra-loop non-training pair: ambiguous ordering, no GT label
                items.append({
                    "prompt":       prompt_fn(hierarchy[i], hierarchy[j],
                                             anchor_winner, anchor_loser),
                    "label":        None,
                    "team_a":       hierarchy[i],
                    "team_b":       hierarchy[j],
                    "rank_a":       i,
                    "rank_b":       j,
                    "gap":          gap,
                    "is_train":     False,
                    "is_exception": False,
                    "no_gt":        True,
                    "direction":    "forward",
                })
                items.append({
                    "prompt":       prompt_fn(hierarchy[j], hierarchy[i],
                                             anchor_winner, anchor_loser),
                    "label":        None,
                    "team_a":       hierarchy[j],
                    "team_b":       hierarchy[i],
                    "rank_a":       j,
                    "rank_b":       i,
                    "gap":          gap,
                    "is_train":     False,
                    "is_exception": False,
                    "no_gt":        True,
                    "direction":    "backward",
                })
            else:
                # Normal pair: original ordering is ground truth
                items.append({
                    "prompt":       prompt_fn(hierarchy[i], hierarchy[j],
                                             anchor_winner, anchor_loser),
                    "label":        "Yes",
                    "team_a":       hierarchy[i],
                    "team_b":       hierarchy[j],
                    "rank_a":       i,
                    "rank_b":       j,
                    "gap":          gap,
                    "is_train":     is_adj,
                    "is_exception": False,
                    "no_gt":        False,
                    "direction":    "forward",
                })
                items.append({
                    "prompt":       prompt_fn(hierarchy[j], hierarchy[i],
                                             anchor_winner, anchor_loser),
                    "label":        "No",
                    "team_a":       hierarchy[j],
                    "team_b":       hierarchy[i],
                    "rank_a":       j,
                    "rank_b":       i,
                    "gap":          gap,
                    "is_train":     is_adj,
                    "is_exception": False,
                    "no_gt":        False,
                    "direction":    "backward",
                })
    return items


# ═══════════════════════════════════════════════════════════════
# Model utilities
# ═══════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_name: str, local_files_only: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tokenizer


def get_yes_no_token_ids(tokenizer) -> tuple[int, int]:
    """
    Find the single-token IDs for "Yes" and "No".

    We prefer the space-prefixed variants (" Yes", " No") because causal
    LMs typically expect a leading space after "A:".  Falls back gracefully
    if none of the candidates are single tokens.
    """
    candidates_yes = [" Yes", "Yes", " yes", "yes"]
    candidates_no  = [" No",  "No",  " no",  "no"]

    def first_single_token(candidates):
        for c in candidates:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        # Last resort: take the first sub-token of the first candidate
        return tokenizer.encode(candidates[0], add_special_tokens=False)[0]

    return first_single_token(candidates_yes), first_single_token(candidates_no)


@torch.no_grad()
def evaluate_items(
    model,
    tokenizer,
    items: list[dict],
    yes_id: int,
    no_id: int,
) -> list[dict]:
    """
    For each item, compute:
      logit_yes, logit_no   — raw vocab logits at the final token position
      logit_diff            — logit_yes - logit_no  (positive = model prefers Yes)
      prob_yes, prob_no     — softmax renormalised over {Yes, No} only
    """
    results = []
    for item in items:
        input_ids = tokenizer.encode(item["prompt"], return_tensors="pt").to(model.device)
        output = model(input_ids)
        last_logits = output.logits[0, -1, :]
        logit_yes = last_logits[yes_id].item()
        logit_no  = last_logits[no_id].item()

        logit_diff = logit_yes - logit_no
        log_probs = F.log_softmax(torch.tensor([logit_yes, logit_no]), dim=0)
        prob_yes  = log_probs[0].exp().item()
        prob_no   = log_probs[1].exp().item()

        results.append({
            **item,
            "logit_yes":  logit_yes,
            "logit_no":   logit_no,
            "logit_diff": logit_diff,
            "prob_yes":   prob_yes,
            "prob_no":    prob_no,
        })
    return results


@torch.no_grad()
def extract_representations(
    model,
    tokenizer,
    items: list[dict],
) -> np.ndarray:
    """
    Extract the final-layer hidden state at the last token position for each item.
    This is the vector immediately preceding lm_head — the same representation
    that lm_head reads out to produce Yes/No logits.

    Returns float32 array of shape (len(items), hidden_size).
    """
    reps = []
    for item in items:
        input_ids = tokenizer.encode(item["prompt"], return_tensors="pt").to(model.device)
        output = model(input_ids, output_hidden_states=True)
        # hidden_states[-1]: final transformer layer output, shape (1, seq_len, hidden_size)
        vec = output.hidden_states[-1][0, -1, :].float().cpu().numpy()
        reps.append(vec)
    return np.stack(reps)  # (n_items, hidden_size)


def compute_rep_similarity_matrix(reps: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute the (n_items, n_items) pairwise similarity matrix.
    metric: 'cosine' (L2-normalised dot product) or 'dot' (raw inner product).
    """
    if metric == "cosine":
        norms = np.linalg.norm(reps, axis=1, keepdims=True)
        reps = reps / np.maximum(norms, 1e-8)
    return (reps @ reps.T).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Finetuning — shared helper
# ═══════════════════════════════════════════════════════════════

def _label_to_id(label: str, yes_id: int, no_id: int) -> int:
    return yes_id if label == "Yes" else no_id


def _make_optimizer(params, method: str, optimizer_type: str, lr: float):
    cls = torch.optim.SGD if optimizer_type == "sgd" else torch.optim.AdamW
    return cls(params, lr=lr)


def _gradient_step(model, input_ids, label_id: int, optimizer) -> float:
    """Single supervised gradient step on the last-token logit."""
    optimizer.zero_grad()
    output = model(input_ids)
    last_logits = output.logits[0, -1, :]
    loss = F.cross_entropy(
        last_logits.unsqueeze(0),
        torch.tensor([label_id], device=model.device),
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def _gradient_step_batch(
    model,
    tokenizer,
    items: list[dict],
    yes_id: int,
    no_id: int,
    optimizer,
    l2_reg: float = 0.0,
    ref_weight=None,
) -> float:
    """
    Batched supervised gradient step.

    Sequences are left-padded to the same length so that the last real token
    is always at position -1, regardless of sequence length.  This lets us
    read all label logits from a single output[:, -1, :] slice.

    l2_reg / ref_weight: if l2_reg > 0 and ref_weight is not None, adds
      l2_reg * ||lm_head.weight - ref_weight||^2 to the loss.  Only meaningful
      when finetuning lm_head (probing method).
    """
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    encodings = [tokenizer.encode(item["prompt"], add_special_tokens=True) for item in items]
    max_len = max(len(e) for e in encodings)

    # Left-pad: last real token is always at index -1 after padding
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(e)) + e for e in encodings],
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = (input_ids != pad_id).long()

    label_ids = torch.tensor(
        [_label_to_id(item["label"], yes_id, no_id) for item in items],
        device=model.device,
    )

    optimizer.zero_grad()
    output = model(input_ids, attention_mask=attention_mask)
    last_logits = output.logits[:, -1, :]  # (batch, vocab)
    loss = F.cross_entropy(last_logits, label_ids)
    if l2_reg > 0.0 and ref_weight is not None:
        loss = loss + l2_reg * (model.lm_head.weight - ref_weight).pow(2).sum()
    loss.backward()
    optimizer.step()
    return loss.item()


# ═══════════════════════════════════════════════════════════════
# Finetuning — full
# ═══════════════════════════════════════════════════════════════

def finetune_full(
    model,
    tokenizer,
    items: list[dict],
    yes_id: int,
    no_id: int,
    lr: float        = 1e-5,
    n_epochs: int    = 1,
    max_steps: int   = -1,
    batch_size: int  = 16,
    shuffle: bool    = False,
    optimizer_type: str = "adamw",
) -> None:
    """
    Full-parameter fine-tune on all training items.

    SUGGESTION: Start with n_epochs=1 and a very small lr (1e-5 or lower) —
    these models are already calibrated and large learning rates catastrophically
    forget the base distribution within a few steps.
    """
    train_items = [it for it in items if it["is_train"]]
    optimizer = _make_optimizer(model.parameters(), "full", optimizer_type, lr)
    model.train()
    step = 0
    for epoch in range(n_epochs):
        batch = list(train_items)
        if shuffle:
            random.shuffle(batch)
        for i in range(0, len(batch), batch_size):
            if max_steps >= 0 and step >= max_steps:
                break
            chunk = batch[i:i + batch_size]
            loss = _gradient_step_batch(model, tokenizer, chunk, yes_id, no_id, optimizer)
            step += 1
            print(f"  [full] epoch={epoch+1} step={step} loss={loss:.4f}")
        else:
            continue
        break  # max_steps reached
    model.eval()


# ═══════════════════════════════════════════════════════════════
# Finetuning — LoRA
# ═══════════════════════════════════════════════════════════════

def finetune_lora(
    model,
    tokenizer,
    items: list[dict],
    yes_id: int,
    no_id: int,
    lr: float        = 1e-4,
    n_epochs: int    = 1,
    max_steps: int   = -1,
    batch_size: int  = 16,
    lora_r: int      = 8,
    lora_alpha: int  = 16,
    target_modules   = None,
    shuffle: bool    = False,
    optimizer_type: str = "adamw",
):
    """
    LoRA fine-tune via peft.

    SUGGESTIONS:
    - lora_r=4 is often sufficient for tiny datasets (fewer free parameters,
      less overfitting).
    - target_modules defaults to ["q_proj", "v_proj"] which is a common
      minimal choice.  Adding "k_proj" and "o_proj" sometimes helps with
      relational tasks.
    - lora_alpha=2*lora_r (the default here) sets the effective LR scaling
      to 2.0 regardless of r — a reasonable neutral starting point.

    Returns the peft-wrapped model (the caller must use this going forward).
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError("peft is required for LoRA finetuning: pip install peft")

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_items = [it for it in items if it["is_train"]]
    optimizer = _make_optimizer(model.parameters(), "lora", optimizer_type, lr)
    model.train()
    step = 0
    for epoch in range(n_epochs):
        batch = list(train_items)
        if shuffle:
            random.shuffle(batch)
        for i in range(0, len(batch), batch_size):
            if max_steps >= 0 and step >= max_steps:
                break
            chunk = batch[i:i + batch_size]
            loss = _gradient_step_batch(model, tokenizer, chunk, yes_id, no_id, optimizer)
            step += 1
            print(f"  [lora] epoch={epoch+1} step={step} loss={loss:.4f}")
        else:
            continue
        break
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# Finetuning — sequential per-example SGD
# ═══════════════════════════════════════════════════════════════

def finetune_sgd_sequential(
    model,
    tokenizer,
    items: list[dict],
    yes_id: int,
    no_id: int,
    lr: float              = 1e-4,
    n_steps_per_item: int  = 1,
    max_steps: int         = -1,
    batch_size: int        = 16,
    shuffle: bool          = False,
    optimizer_type: str    = "adamw",
) -> None:
    """
    Online / per-example finetuning.

    Training items are visited one at a time (in hierarchy order by default),
    and n_steps_per_item gradient steps are taken for each.

    SUGGESTIONS:
    - Start with n_steps=1 to simulate the minimal in-context gradient update.
    - Curriculum order (A→B then B→C, etc.) is the default.  Use --shuffle-train
      to test whether order matters.
    - This is the most informative method for studying how knowledge about
      one pair "bleeds into" adjacent pairs after a minimal update.
    - The optimizer is re-used across items (state accumulates), which means
      AdamW's moment estimates carry information from earlier items.  This is
      realistic but worth noting.  Use --sgd-optimizer sgd for a stateless
      alternative.
    """
    train_items = [it for it in items if it["is_train"]]

    # Default curriculum: sort by the minimum rank index in the pair so we
    # move through the hierarchy in order.  Within each adjacency, visit
    # forward before backward.
    if not shuffle:
        train_items = sorted(
            train_items,
            key=lambda x: (min(x["rank_a"], x["rank_b"]), x["direction"] != "forward"),
        )
    else:
        random.shuffle(train_items)

    optimizer = _make_optimizer(model.parameters(), "sgd_sequential", optimizer_type, lr)
    model.train()
    total_step = 0
    for i in range(0, len(train_items), batch_size):
        if max_steps >= 0 and total_step >= max_steps:
            break
        chunk = train_items[i:i + batch_size]
        chunk_label = f"items {i+1}-{i+len(chunk)}"
        for s in range(n_steps_per_item):
            if max_steps >= 0 and total_step >= max_steps:
                break
            loss = _gradient_step_batch(model, tokenizer, chunk, yes_id, no_id, optimizer)
            total_step += 1
            print(f"  [seq_sgd] {chunk_label} step={s+1}/{n_steps_per_item} loss={loss:.4f}")
    model.eval()


# ═══════════════════════════════════════════════════════════════
# Finetuning — lm_head readout
# ═══════════════════════════════════════════════════════════════

def finetune_probing(
    model,
    tokenizer,
    items: list[dict],
    yes_id: int,
    no_id: int,
    lr: float       = 1e-3,
    n_epochs: int   = 50,
    batch_size: int = 16,
    shuffle: bool   = False,
    l2_reg: float   = 0.0,
) -> None:
    """
    Linear readout method: freeze the entire model except for lm_head, then
    fine-tune lm_head on the training items using the model's own vocab logits.

    Because only lm_head is updated the hidden representations are fixed, so
    this isolates whether the information needed for the hierarchy is already
    present in the final hidden states and just needs to be read out differently.

    Post-training evaluation uses evaluate_items as normal (no special probe
    object needed — lm_head changes are in place on the model).

    l2_reg: if > 0, adds l2_reg * ||W - W_0||^2 to the loss at every step,
      where W_0 is the lm_head weight at the start of training.  This regularises
      the readout towards its initialisation, limiting how far it can drift.

    SUGGESTIONS:
    - Use a higher lr (1e-3) than other methods since lm_head is just one
      linear layer and converges quickly.
    - n_epochs 20–100 is typical; loss usually plateaus within 20 epochs on
      these tiny datasets.
    - Compare against sgd_sequential to dissect how much of the effect comes
      from updating lm_head vs the rest of the network.
    """
    # Freeze everything; selectively unfreeze lm_head
    for name, param in model.named_parameters():
        param.requires_grad = "lm_head" in name
    for param in model.lm_head.parameters():
        param.requires_grad = True

    trainable = model.lm_head.parameters()
    if not trainable:
        raise RuntimeError(
            "No parameters found matching 'lm_head'. "
            "Check model architecture (lm_head name may differ)."
        )
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    # Snapshot initial weights for L2 regularisation (only if needed)
    ref_weight = model.lm_head.weight.data.clone().detach() if l2_reg > 0.0 else None

    model.train()

    train_items = [it for it in items if it["is_train"]]

    for epoch in range(n_epochs):
        batch = list(train_items)
        if shuffle:
            random.shuffle(batch)
        epoch_loss = 0.0
        n_chunks = 0
        for i in range(0, len(batch), batch_size):
            chunk = batch[i:i + batch_size]
            loss = _gradient_step_batch(
                model, tokenizer, chunk, yes_id, no_id, optimizer,
                l2_reg=l2_reg, ref_weight=ref_weight,
            )
            epoch_loss += loss
            n_chunks += 1
        if (epoch + 1) % 10 == 0:
            print(f"  [probing] epoch={epoch+1}/{n_epochs} avg_loss={epoch_loss/n_chunks:.4f}")

    model.eval()
    # Restore all params to require_grad for consistent post-eval
    for param in model.parameters():
        param.requires_grad = True


# ═══════════════════════════════════════════════════════════════
# Results & display
# ═══════════════════════════════════════════════════════════════

def _build_paired_correct(results: list[dict]) -> dict:
    """
    For each unordered pair {A, B} where A > B in the hierarchy, the model
    is correct iff prob_yes(forward: A beats B) > prob_yes(backward: B beats A).

    Items with no_gt=True are excluded (their ordering is ambiguous).

    Returns a dict mapping (min_rank, max_rank) -> bool.
    """
    by_key: dict = {}
    for r in results:
        if r.get("no_gt", False):
            continue
        key = (min(r["rank_a"], r["rank_b"]), max(r["rank_a"], r["rank_b"]))
        by_key.setdefault(key, {})[r["direction"]] = r

    paired_correct = {}
    for key, pair in by_key.items():
        if "forward" in pair and "backward" in pair:
            paired_correct[key] = pair["forward"]["prob_yes"] > pair["backward"]["prob_yes"]
    return paired_correct


def print_results_table(pre: list[dict], post: list[dict]) -> None:
    """Print a side-by-side pre/post comparison table."""
    header = (
        f"{'Pair':<48} {'Split':<6} {'Dir':<5} {'Label':<6} "
        f"{'Pre ΔLogit':>10} {'Post ΔLogit':>11} {'Δ(ΔLogit)':>10} "
        f"{'Pre P(Y)':>9} {'Post P(Y)':>9} {'Paired':>8}"
    )
    print("\n" + header)
    print("─" * len(header))

    pre_correct  = _build_paired_correct(pre)
    post_correct = _build_paired_correct(post)

    for p, q in zip(pre, post):
        pair    = f"{p['team_a'][:22]} vs {p['team_b'][:22]}"
        split   = "TRAIN" if p["is_train"] else "TEST "
        direction = p["direction"][:3]
        label   = p["label"] if p["label"] is not None else "N/A"
        key = (min(p["rank_a"], p["rank_b"]), max(p["rank_a"], p["rank_b"]))
        pre_ok  = pre_correct.get(key, False)
        post_ok = post_correct.get(key, False)
        delta_logit_diff = q["logit_diff"] - p["logit_diff"]
        print(
            f"{pair:<48} {split:<6} {direction:<5} {label:<6} "
            f"{p['logit_diff']:>10.3f} {q['logit_diff']:>11.3f} {delta_logit_diff:>+10.3f} "
            f"{p['prob_yes']:>9.4f} {q['prob_yes']:>9.4f} "
            f"{'✓' if pre_ok else '✗'}→{'✓' if post_ok else '✗'}"
        )


def print_gap_summary(pre: list[dict], post: list[dict]) -> None:
    """
    Aggregate results by gap (adjacent=1, transitive=2, 3, …).
    Useful for quickly seeing whether transitivity generalises.
    """
    from collections import defaultdict
    by_gap_pre  = defaultdict(list)
    by_gap_post = defaultdict(list)
    for p, q in zip(pre, post):
        by_gap_pre[p["gap"]].append(p)
        by_gap_post[p["gap"]].append(q)

    # Paired accuracy: correct iff prob_yes(A beats B) > prob_yes(B beats A)
    # for each unordered pair {A, B} where A > B.
    pre_correct_all  = _build_paired_correct([x for xs in by_gap_pre.values()  for x in xs])
    post_correct_all = _build_paired_correct([x for xs in by_gap_post.values() for x in xs])

    print("\n── Gap summary (forward logit_diff avg; paired accuracy) ──")
    print(f"{'Gap':>4}  {'Split':>6}  {'Pre ΔLogit':>10}  {'Post ΔLogit':>11}  {'Δ(ΔLogit)':>10}  {'Pre Acc':>8}  {'Post Acc':>9}")
    for gap in sorted(by_gap_pre):
        split_label = "TRAIN" if gap == 1 else "TEST "
        # Forward (Yes-labelled) items for logit_diff averages
        pre_fwd  = [x for x in by_gap_pre[gap]  if x["direction"] == "forward"]
        post_fwd = [x for x in by_gap_post[gap] if x["direction"] == "forward"]
        if not pre_fwd:
            continue
        avg_pre  = np.mean([x["logit_diff"] for x in pre_fwd])
        avg_post = np.mean([x["logit_diff"] for x in post_fwd])
        # Paired accuracy: look up each forward pair's correctness
        keys = [(min(x["rank_a"], x["rank_b"]), max(x["rank_a"], x["rank_b"])) for x in pre_fwd]
        acc_pre  = np.mean([1.0 if pre_correct_all.get(k, False)  else 0.0 for k in keys])
        acc_post = np.mean([1.0 if post_correct_all.get(k, False) else 0.0 for k in keys])
        print(
            f"{gap:>4}  {split_label:>6}  {avg_pre:>10.3f}  {avg_post:>11.3f}  "
            f"{avg_post-avg_pre:>+10.3f}  {acc_pre:>8.2%}  {acc_post:>9.2%}"
        )


def save_results(
    output_dir: str,
    hierarchy: list[str],
    pre: list[dict],
    post: list[dict],
    args: argparse.Namespace,
    anchor_winner: str = ANCHOR_WINNER,
    anchor_loser: str  = ANCHOR_LOSER,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "args":          vars(args),
        "hierarchy":     hierarchy,
        "anchor_winner": anchor_winner,
        "anchor_loser":  anchor_loser,
        "pre_finetune":  pre,
        "post_finetune": post,
    }
    out_path = Path(output_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


def save_rep_sim(
    output_dir: str,
    items: list[dict],
    pre_reps: np.ndarray,
    post_reps: np.ndarray,
    metric: str,
) -> None:
    """
    Compute pairwise similarity matrices from pre- and post-finetuning representations
    and save them to disk.

    Saves:
      rep_sim_pre.npy   — (n_items, n_items) float32, pre-finetuning
      rep_sim_post.npy  — (n_items, n_items) float32, post-finetuning
      rep_sim_meta.json — row/column metadata (one entry per dataset item)
    """
    os.makedirs(output_dir, exist_ok=True)
    pre_sim  = compute_rep_similarity_matrix(pre_reps,  metric)
    post_sim = compute_rep_similarity_matrix(post_reps, metric)

    item_meta = [
        {
            "index":        idx,
            "team_a":       it["team_a"],
            "team_b":       it["team_b"],
            "rank_a":       it["rank_a"],
            "rank_b":       it["rank_b"],
            "direction":    it["direction"],
            "is_train":     it["is_train"],
            "is_exception": it.get("is_exception", False),
            "no_gt":        it.get("no_gt", False),
            "label":        it["label"],
            "gap":          it["gap"],
        }
        for idx, it in enumerate(items)
    ]

    np.save(Path(output_dir) / "rep_sim_pre.npy",  pre_sim)
    np.save(Path(output_dir) / "rep_sim_post.npy", post_sim)
    with open(Path(output_dir) / "rep_sim_meta.json", "w") as f:
        json.dump({"metric": metric, "n_items": len(items), "items": item_meta}, f, indent=2)
    print(
        f"Rep-sim matrices ({metric}, {len(items)}\u00d7{len(items)}) saved \u2192 "
        f"{output_dir}/rep_sim_{{pre,post}}.npy + rep_sim_meta.json"
    )


# ═══════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sports hierarchy finetuning experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Dataset ---
    g = p.add_argument_group("Dataset")
    g.add_argument("--task", choices=["sports", "poker_equity"], default="sports",
                   help="Which hierarchy task to run (sports = fake/real MLB teams; "
                        "poker_equity = pre-flop heads-up hands from equity matrix)")
    g.add_argument("--team-type",  choices=["real", "fake"], default="fake",
                   help="Use real MLB teams or plausible fake team names (sports task only)")
    g.add_argument("--n",          type=int, default=7,
                   help="Hierarchy depth (number of teams/hands)")
    g.add_argument("--reverse",    action="store_true",
                   help="Reverse the hierarchy (counterbalancing; flips all labels)")
    g.add_argument("--shuffle-train", action="store_true",
                   help="Shuffle training items instead of visiting them in hierarchy order")
    g.add_argument("--exception", nargs=2, type=int, metavar=("WINNER_RANK", "LOSER_RANK"),
                   default=None,
                   help="Add exception training pair: WINNER_RANK beats LOSER_RANK despite "
                        "being weaker in the original hierarchy (WINNER_RANK > LOSER_RANK)")

    # --- Poker equity ---
    g = p.add_argument_group("Poker equity (--task poker_equity)")
    g.add_argument("--poker-equity-path", default="results/poker_equity/preflop_equity.npy",
                   help="Path to preflop_equity.npy produced by compute_equity_batch.py")
    g.add_argument("--poker-low",  type=float, default=0.51,
                   help="Lower bound on equity for a valid hierarchy edge (hand_a beats hand_b "
                        "with equity in [poker-low, poker-high])")
    g.add_argument("--poker-high", type=float, default=0.60,
                   help="Upper bound on equity for a valid hierarchy edge (keeps edges close so "
                        "hands are comparable; use 1.0 to allow dominant hands)")

    # --- Model ---
    g = p.add_argument_group("Model")
    g.add_argument("--model", default="Qwen/Qwen3.5-4B-Base",
                   help="HuggingFace model name or local path")
    g.add_argument("--local-files-only", action="store_true", default=True,
                   help="Load model from local HF cache only (no network requests)")
    g.add_argument("--no-local-files-only", action="store_false", dest="local_files_only",
                   help="Allow downloading model from HuggingFace if not cached")

    # --- Finetuning ---
    g = p.add_argument_group("Finetuning")
    g.add_argument("--finetune-method",
                   choices=["full", "lora", "sgd_sequential", "probing", "none"],
                   default="sgd_sequential")
    g.add_argument("--lr",             type=float, default=1e-4,
                   help="Learning rate")
    g.add_argument("--n-epochs",       type=int,   default=1,
                   help="Training epochs (full / lora / probing)")
    g.add_argument("--n-steps",        type=int,   default=1,
                   help="Gradient steps per item (sgd_sequential)")
    g.add_argument("--max-steps",      type=int,   default=-1,
                   help="Global gradient step cap across all methods (-1 = unlimited)")
    g.add_argument("--batch-size",     type=int,   default=16,
                   help="Number of training items per gradient step")
    g.add_argument("--sgd-optimizer",  choices=["adamw", "sgd"], default="adamw",
                   help="Optimizer for all methods (sgd = stateless gradient descent)")

    # --- LoRA ---
    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r",      type=int, default=8)
    g.add_argument("--lora-alpha",  type=int, default=16,
                   help="LoRA alpha; effective LR scale = alpha/r (default 2×)")
    g.add_argument("--lora-modules", nargs="+", default=["q_proj", "v_proj"],
                   help="Which linear modules to apply LoRA to")

    # --- Probing (lm_head readout) ---
    g = p.add_argument_group("Probing")
    g.add_argument("--probe-epochs", type=int, default=50,
                   help="Epochs for training the lm_head readout (probing method)")
    g.add_argument("--probe-l2-reg", type=float, default=0.0,
                   help="L2 regularisation weight penalising the squared distance of lm_head "
                        "weights from their initialisation (probing method only; 0 = disabled)")

    # --- Output ---
    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", default="results",
                   help="Directory for results.json")
    g.add_argument("--seed",       type=int, default=42)

    # --- Representation similarity ---
    g = p.add_argument_group("Representation similarity")
    g.add_argument("--save-rep-sim", action="store_true",
                   help="Extract final-layer hidden states and save pairwise similarity matrices "
                        "(n_items \u00d7 n_items) pre- and post-finetuning. "
                        "Saved to rep_sim_{pre,post}.npy + rep_sim_meta.json in --output-dir.")
    g.add_argument("--rep-sim-metric", choices=["cosine", "dot"], default="cosine",
                   help="Similarity metric for --save-rep-sim (default: cosine)")
    g.add_argument("--rep-sim-only", action="store_true",
                   help="Skip finetuning and only extract the representation similarity matrix "
                        "(implies --save-rep-sim and --finetune-method none)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    if args.rep_sim_only:
        args.save_rep_sim = True
        args.finetune_method = "none"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── 1. Build hierarchy ──────────────────────────────────────
    exception = tuple(args.exception) if args.exception is not None else None

    if args.task == "poker_equity":
        print("=" * 60)
        print("Task: poker_equity (pre-flop all-in heads-up)")
        print(f"Loading equity data from: {args.poker_equity_path}")
        matrix, labels = load_poker_equity_data(args.poker_equity_path)
        rng = random.Random(args.seed)
        hierarchy = find_poker_hierarchy(
            matrix, labels,
            n=args.n,
            low=args.poker_low,
            high=args.poker_high,
            rng=rng,
            exception=exception,
            exclude=[POKER_ANCHOR_WINNER, POKER_ANCHOR_LOSER],
        )
        hierarchy = build_hierarchy(hierarchy, reverse=args.reverse)
        anchor_winner = POKER_ANCHOR_WINNER
        anchor_loser  = POKER_ANCHOR_LOSER
        prompt_fn     = make_prompt_poker
        print(f"Hierarchy ({'reversed' if args.reverse else 'forward'}, poker hands):")
        for i, h in enumerate(hierarchy):
            print(f"  rank {i}: {h}")
        print(f"\nAnchor:  {anchor_winner} > {anchor_loser}")

    else:  # sports (default)
        # Sample anchor pair randomly from real MLB teams (always real, regardless of team_type)
        anchor_winner, anchor_loser = random.sample(REAL_MLB_TEAMS, 2)
        pool = (
            [t for t in REAL_MLB_TEAMS if t not in (anchor_winner, anchor_loser)]
            if args.team_type == "real"
            else list(FAKE_MLB_TEAMS)
        )
        if len(pool) < args.n:
            raise ValueError(
                f"Not enough teams in pool ({len(pool)}) for hierarchy of depth {args.n}. "
                f"Reduce --n or extend the team list."
            )
        sampled   = random.sample(pool, args.n)
        hierarchy = build_hierarchy(sampled, reverse=args.reverse)
        prompt_fn = make_prompt
        print("=" * 60)
        print(f"Hierarchy ({'reversed' if args.reverse else 'forward'}, {args.team_type} teams):")
        for i, t in enumerate(hierarchy):
            print(f"  rank {i}: {t}")
        print(f"\nAnchor:  {anchor_winner} > {anchor_loser}")

    # ── 2. Generate dataset ─────────────────────────────────────
    if exception is not None:
        exc_w, exc_l = exception
        print(f"\nException: rank {exc_w} beats rank {exc_l} (inconsistent with hierarchy)")
    dataset     = generate_dataset(
        hierarchy,
        anchor_winner=anchor_winner,
        anchor_loser=anchor_loser,
        exception=exception,
        prompt_fn=prompt_fn,
    )
    train_items = [x for x in dataset if x["is_train"]]
    test_items  = [x for x in dataset if not x["is_train"] and not x.get("no_gt", False)]
    no_gt_items = [x for x in dataset if x.get("no_gt", False)]
    print(
        f"\nDataset: {len(train_items)} train items (adjacent pairs + exception, both directions), "
        f"{len(test_items)} test items (transitive pairs), "
        f"{len(no_gt_items)} no-GT items (ambiguous intra-loop pairs)"
    )

    # ── 3. Load model ───────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, local_files_only=args.local_files_only)
    yes_id, no_id    = get_yes_no_token_ids(tokenizer)
    print(
        f"Yes token: {tokenizer.decode([yes_id])!r} (id={yes_id}),  "
        f"No  token: {tokenizer.decode([no_id])!r} (id={no_id})"
    )

    # ── 4. Pre-finetuning evaluation ────────────────────────────
    print("\nEvaluating (pre-finetuning)…")
    pre_results = evaluate_items(model, tokenizer, dataset, yes_id, no_id)
    if args.save_rep_sim:
        print("Extracting representations (pre-finetuning)…")
        pre_reps = extract_representations(model, tokenizer, dataset)

    # ── 5. Finetuning ───────────────────────────────────────────
    if args.finetune_method == "none":
        print("\nSkipping finetuning (--finetune-method none).")

    elif args.finetune_method == "full":
        print(f"\nFinetuning (full, lr={args.lr}, epochs={args.n_epochs}, batch={args.batch_size})…")
        finetune_full(
            model, tokenizer, dataset, yes_id, no_id,
            lr=args.lr, n_epochs=args.n_epochs, max_steps=args.max_steps,
            batch_size=args.batch_size, shuffle=args.shuffle_train,
            optimizer_type=args.sgd_optimizer,
        )

    elif args.finetune_method == "lora":
        print(f"\nFinetuning (LoRA r={args.lora_r} α={args.lora_alpha}, lr={args.lr}, epochs={args.n_epochs}, batch={args.batch_size})…")
        model = finetune_lora(
            model, tokenizer, dataset, yes_id, no_id,
            lr=args.lr, n_epochs=args.n_epochs, max_steps=args.max_steps,
            batch_size=args.batch_size, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            shuffle=args.shuffle_train, optimizer_type=args.sgd_optimizer,
        )

    elif args.finetune_method == "sgd_sequential":
        print(
            f"\nFinetuning (sequential, {args.n_steps} step(s)/batch, "
            f"batch={args.batch_size}, lr={args.lr}, optimizer={args.sgd_optimizer})…"
        )
        finetune_sgd_sequential(
            model, tokenizer, dataset, yes_id, no_id,
            lr=args.lr, n_steps_per_item=args.n_steps, max_steps=args.max_steps,
            batch_size=args.batch_size, shuffle=args.shuffle_train,
            optimizer_type=args.sgd_optimizer,
        )

    elif args.finetune_method == "probing":
        print(f"\nFinetuning lm_head readout (lr={args.lr}, epochs={args.probe_epochs}, batch={args.batch_size}, l2_reg={args.probe_l2_reg})…")
        finetune_probing(
            model, tokenizer, dataset, yes_id, no_id,
            lr=args.lr, n_epochs=args.probe_epochs,
            batch_size=args.batch_size, shuffle=args.shuffle_train,
            l2_reg=args.probe_l2_reg,
        )

    # ── 6. Post-finetuning evaluation (always runs) ─────────────
    print("\nEvaluating (post-finetuning)…")
    post_results = evaluate_items(model, tokenizer, dataset, yes_id, no_id)
    if args.save_rep_sim:
        print("Extracting representations (post-finetuning)…")
        post_reps = extract_representations(model, tokenizer, dataset)
        save_rep_sim(args.output_dir, dataset, pre_reps, post_reps, args.rep_sim_metric)

    # ── 7. Display & save ───────────────────────────────────────
    print_results_table(pre_results, post_results)
    print_gap_summary(pre_results, post_results)
    save_results(args.output_dir, hierarchy, pre_results, post_results, args,
                 anchor_winner=anchor_winner, anchor_loser=anchor_loser)


if __name__ == "__main__":
    main()
