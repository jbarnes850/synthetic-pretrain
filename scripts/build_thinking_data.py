#!/usr/bin/env python3
"""Build interleaved-thinking mid-training data from raw pretraining chunks."""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from common import jsonl_iter, load_config, resolve_hf_path, set_seed, write_jsonl
from transformers import AutoTokenizer

try:
    import requests
except Exception:  # pragma: no cover - dependency guard
    requests = None


PROMPT = """Below is text scraped from a web page. The text does not contain all
implicit contexts which are well known to the author, such as world knowledge,
commonsense, the author's internal thoughts, goals and preferences, etc. Your
task is to augment the text to add missing contexts and actions, so that the
augmented text should imitate how an intelligent learner is actively reasoning
and taking actions to understand the text and predict what comes next.
Importantly, the added actions should demonstrate meta-learning skills, e.g.
proactively self-reflect and distill lessons so as to maximize accuracy and
speed of predicting the future, especially generalize to unseen and different
tasks.

First, reconstruct the global context. Such global context should provide
background on how the text was generated. For example, identify who wrote the
text, their goal(s), and the relevant world model(s) need to be recalled, such
as common knowledge, commonsense, common logical rules, causal relations,
reasoning strategies, physical and social principles etc., as well as knowledge
and logical rules, and key reasoning steps specific to this text. The global
context should also copy details which are specific to the text and would
otherwise be almost impossible for anyone to predict without seeing them in the
global context, e.g. dates, names, text from web scraping, etc. Put the global
context between <global_context> and </global_context>. DO NOT mention "user".

Second, insert {min_thoughts}-{max_thoughts} total missing intermediate contexts
and actions in the interleaving fashion, with the same goal of reconstructing
context needed to predict the subsequent text. To help teach meta-learning
skills, the reconstructed missing context should demonstrate how an intelligent
human will make sense of the text, such as reconstruct a world model with
physics or social principles that can predict dynamics of the scenario, as well
as derive or infer implications specific to the matters in the text, etc. For
example, the inserted context should reconstruct agent(s) in the text and all
the implicit agentic capabilities they took, such as planning, reasoning,
metacognition, reflection, tool use, etc. that had resulted in the text. You
should find the highest-leverage agentic and meta-reasoning strategies the human
agent(s) may have used but not explicitly written in the text.

The inserted context should be from the first-person perspective of the agent
who wrote the text (if there are multiple agents, first stating which agent this
first-person perspective is from). To make the agentic capabilities more
explicit, organize the inserted context with specific action tags, such as:
- <task> ... </task> or <set_goal> ... </set_goal> for making any implicit goal
  or preference more concrete and explicit so as to provide context for
  subsequent actions.
- <think> ... </think> for reconstructing the inner monologues that will lead
  to the agent producing or understanding the text, including various reasoning
  skills such as induction, deduction, abduction, counterfactual reasoning,
  logical reasoning, causal reasoning, probabilistic reasoning, constraints
  satisfaction, planning, etc. and meta-reasoning strategies illustrated above.
- <world_model> ... </world_model> to reconstruct a self-contained world which
  can simulate the events in the text, such as detailed background knowledge, a
  set of generally-true facts, physical and social principles, and commonsense
  that drive the changes of states in the world after agent(s) take different
  actions.
- <recall_knowledge> ... </recall_knowledge> to self-ask and retrieve relevant
  knowledge.
- <simulate> ... </simulate> to simulate possible states of future, different
  outcomes via counterfactual reasoning which would help for predicting
  subsequent actions and events.
- <reusable_lessons> ... </reusable_lessons> for reading and writing a self-note
  which contains reusable abstractions and lessons distilled from learning
  experience.
- <verification> ... </verification> for proactive self-verification and
  reflective reasoning processes.
- <tool_use> ... </tool_use> for invoking external tools, e.g. <python> ...
  </python> for writing python code, <web_search> ... </web_search> for
  browsing the web to check facts, etc.

IMPORTANT: DO NOT change the original text. Preserve all original text content
and order exactly, and only insert tagged contexts between spans of the original
text. Do not summarize, rewrite, omit, answer as an assistant, include markdown
fences, or include a separate explanation. Return only the augmented text.

Text:
{chunk}
"""

ACTION_TAGS = (
    "global_context",
    "think",
    "world_model",
    "recall_knowledge",
    "simulate",
    "simulation",
    "reusable_lessons",
    "verification",
    "task",
    "set_goal",
    "tool_use",
    "python",
    "web_search",
)


def parse_augmented_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"teacher returned non-string content: {type(text).__name__}")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            value = obj.get("augmented_text", "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass
    match = re.search(r'"augmented_text"\s*:\s*"(?P<value>.*)"\s*}', text, flags=re.DOTALL)
    if match:
        try:
            return json.loads('"' + match.group("value") + '"').strip()
        except Exception:
            return match.group("value").replace(r"\/", "/").strip()
    return text.strip()


def normalize_think_tags(text: str) -> str:
    normalized = text
    for tag in ACTION_TAGS:
        normalized = re.sub(fr"<\s*{tag}\s*>", "<think>", normalized, flags=re.IGNORECASE)
        normalized = re.sub(fr"<\s*/\s*{tag}\s*>", "</think>", normalized, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def strip_think_tags(text: str) -> str:
    return re.sub(r"</?think>", "", text).strip()


def remove_think_spans(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def canonical_original_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def prune_think_spans(text: str, max_thoughts: int) -> str:
    spans_seen = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal spans_seen
        spans_seen += 1
        return match.group(0) if spans_seen <= max_thoughts else ""

    return re.sub(r"<think>.*?</think>", repl, text, flags=re.IGNORECASE | re.DOTALL)


def extract_think_spans(text: str, max_thoughts: int) -> list[str]:
    normalized = normalize_think_tags(text)
    spans = re.findall(r"<think>.*?</think>", normalized, flags=re.IGNORECASE | re.DOTALL)
    return [span.strip() for span in spans[:max_thoughts] if strip_think_tags(span)]


def split_interleaved_text(text: str) -> tuple[list[str], list[str]]:
    """Return raw-text segments and think spans in alternating order."""
    normalized = normalize_think_tags(text)
    text_segments: list[str] = []
    spans: list[str] = []
    cursor = 0
    for match in re.finditer(r"<think>.*?</think>", normalized, flags=re.IGNORECASE | re.DOTALL):
        text_segments.append(normalized[cursor : match.start()])
        spans.append(match.group(0).strip())
        cursor = match.end()
    text_segments.append(normalized[cursor:])
    return text_segments, spans


def normalize_for_alignment(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace while keeping a map from normalized chars to original offsets."""
    out: list[str] = []
    norm_to_orig_end: list[int] = []
    in_space = False
    for idx, char in enumerate(unicodedata.normalize("NFKC", text)):
        if char.isspace():
            if not in_space:
                out.append(" ")
                norm_to_orig_end.append(idx + 1)
                in_space = True
            else:
                norm_to_orig_end[-1] = idx + 1
            continue
        out.append(char)
        norm_to_orig_end.append(idx + 1)
        in_space = False
    return "".join(out).strip(), norm_to_orig_end


def normalized_prefix_len(text: str, boundary: int) -> int:
    normalized, _ = normalize_for_alignment(text[:boundary])
    return len(normalized)


def line_around(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, max(0, pos)) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return text[start:end]


def is_inside_fenced_block(text: str, pos: int) -> bool:
    return len(re.findall(r"```", text[:pos])) % 2 == 1


def looks_code_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("```", ">>>", "...", "$ ")):
        return True
    if line.startswith(("    ", "\t")):
        return True
    code_markers = ("<-", "=>", "==", "!=", "::", "{", "}", ";")
    if any(marker in stripped for marker in code_markers):
        return True
    if re.search(r"\b(?:Error|Traceback|Exception)\b", stripped) and re.search(r"[()=,:]", stripped):
        return True
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_.]*\s*=", stripped) and re.search(r"[(),]", stripped):
        return True
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_.]*\s*\([^)]*", stripped) and re.search(r"[=,]", stripped):
        return True
    punctuation = sum(1 for char in stripped if char in "=()[]{}<>;:,")
    return punctuation >= 6 and punctuation / max(1, len(stripped)) > 0.08


def unsafe_code_boundary(text: str, pos: int) -> bool:
    if pos in (0, len(text)):
        return False
    if is_inside_fenced_block(text, pos):
        return True
    current_line = line_around(text, max(0, pos - 1))
    next_line = line_around(text, min(len(text), pos + 1))
    return looks_code_like_line(current_line) or looks_code_like_line(next_line)


def safe_insertion_positions(text: str) -> list[int]:
    positions = {0, len(text)}
    for match in re.finditer(r"\n{2,}", text):
        if not unsafe_code_boundary(text, match.end()):
            positions.add(match.end())
    for match in re.finditer(r"\n", text):
        if not unsafe_code_boundary(text, match.end()):
            positions.add(match.end())
    for match in re.finditer(r"[.!?;:][\"')\]]?\s+", text):
        if not unsafe_code_boundary(text, match.end()):
            positions.add(match.end())
    for match in re.finditer(r"(?:^|\n)(?:#+\s.*|[-*•]\s.*)\n", text):
        if not unsafe_code_boundary(text, match.end()):
            positions.add(match.end())
    return sorted(pos for pos in positions if 0 <= pos <= len(text))


def snap_to_safe_boundary(text: str, target: int, minimum: int = 0, max_distance: int = 360) -> int | None:
    target = max(0, min(len(text), target))
    candidates = [pos for pos in safe_insertion_positions(text) if pos >= minimum]
    if not candidates:
        return None
    best = min(candidates, key=lambda pos: (abs(pos - target), pos < target, pos))
    if best not in (0, len(text)) and abs(best - target) > max_distance:
        return None
    return best


def teacher_boundary_to_raw_boundary(raw_text: str, stripped_teacher: str, teacher_boundary: int) -> int | None:
    raw_norm, raw_map = normalize_for_alignment(raw_text)
    teacher_norm, _ = normalize_for_alignment(stripped_teacher)
    if not raw_norm or not teacher_norm:
        return None
    norm_boundary = normalized_prefix_len(stripped_teacher, teacher_boundary)
    matcher = difflib.SequenceMatcher(None, teacher_norm, raw_norm, autojunk=False)
    blocks = matcher.get_matching_blocks()
    for block in blocks:
        if block.size and block.a <= norm_boundary <= block.a + block.size:
            raw_norm_boundary = block.b + (norm_boundary - block.a)
            break
    else:
        before = [block for block in blocks if block.size and block.a + block.size <= norm_boundary]
        after = [block for block in blocks if block.size and block.a >= norm_boundary]
        candidates: list[tuple[int, int]] = []
        if before:
            block = max(before, key=lambda b: b.a + b.size)
            candidates.append((norm_boundary - (block.a + block.size), block.b + block.size))
        if after:
            block = min(after, key=lambda b: b.a)
            candidates.append((block.a - norm_boundary, block.b))
        if not candidates:
            return None
        raw_norm_boundary = min(candidates, key=lambda item: item[0])[1]
    if raw_norm_boundary <= 0:
        return 0
    if raw_norm_boundary >= len(raw_map):
        return len(raw_text)
    return raw_map[raw_norm_boundary - 1]


def teacher_insertion_boundaries(text: str) -> tuple[str, list[int], list[str]]:
    text_segments, spans = split_interleaved_text(text)
    stripped_parts: list[str] = []
    boundaries: list[int] = []
    cursor = 0
    for index, span in enumerate(spans):
        segment = text_segments[index] if index < len(text_segments) else ""
        stripped_parts.append(segment)
        cursor += len(segment)
        boundaries.append(cursor)
    stripped_parts.extend(text_segments[len(spans) :])
    return "".join(stripped_parts), boundaries, spans


def has_unsafe_thought_boundaries(raw_text: str, augmented_text: str) -> bool:
    stripped_text, boundaries, spans = teacher_insertion_boundaries(augmented_text)
    if not spans:
        return True
    if canonical_original_text(raw_text) != canonical_original_text(stripped_text):
        return True
    min_gap = 80
    interior = [boundary for boundary in boundaries if 0 < boundary < len(stripped_text)]
    min_interior = max(1, min(3, len(spans) // 2))
    if len(interior) < min_interior:
        return True
    if boundaries.count(0) > 1:
        return True
    ordered = sorted(boundaries)
    if any(right - left < min_gap for left, right in zip(ordered, ordered[1:]) if left != right):
        return True
    if len(set(boundaries)) != len(boundaries):
        return True
    safe_positions = set(safe_insertion_positions(stripped_text))
    return any(boundary not in safe_positions for boundary in boundaries)


def alignment_repair_interleaving(raw_text: str, augmented_text: str, max_thoughts: int) -> tuple[str, str] | None:
    stripped_teacher, teacher_boundaries, spans = teacher_insertion_boundaries(augmented_text)
    spans = [span for span in spans[:max_thoughts] if strip_think_tags(span)]
    if not spans:
        return None
    positions: list[int] = []
    minimum = 0
    min_gap = 80
    for index, span in enumerate(spans):
        teacher_boundary = teacher_boundaries[min(index, len(teacher_boundaries) - 1)] if teacher_boundaries else 0
        raw_boundary = teacher_boundary_to_raw_boundary(raw_text, stripped_teacher, teacher_boundary)
        if raw_boundary is None:
            return None
        snapped = snap_to_safe_boundary(raw_text, raw_boundary, minimum=minimum)
        if snapped is None:
            return None
        positions.append(snapped)
        minimum = snapped + min_gap
    return interleave_at_positions(raw_text, spans, positions), "alignment_safe_boundary"


def fallback_safe_boundary_repair(raw_text: str, spans: list[str]) -> tuple[str, str] | None:
    safe_positions = safe_insertion_positions(raw_text)
    if not spans or len(safe_positions) < len(spans):
        return None
    positions = [0]
    usable = [pos for pos in safe_positions if pos not in (0, len(raw_text))]
    min_gap = 80
    for i in range(1, len(spans)):
        if not usable:
            return None
        target = round(len(raw_text) * i / len(spans))
        later = [pos for pos in usable if pos >= positions[-1] + min_gap]
        if not later:
            return None
        best = min(later, key=lambda pos: (abs(pos - target), pos))
        positions.append(best)
        usable = [pos for pos in usable if pos >= best + min_gap]
    return interleave_at_positions(raw_text, spans, positions), "fallback_safe_boundary"


def interleave_at_positions(raw_text: str, spans: list[str], positions: list[int]) -> str:
    parts: list[str] = []
    cursor = 0
    for pos, span in zip(positions, spans, strict=True):
        pos = max(cursor, min(len(raw_text), pos))
        parts.append(raw_text[cursor:pos])
        parts.append(f"\n\n{span}\n\n")
        cursor = pos
    parts.append(raw_text[cursor:])
    return "".join(parts).strip()


def interleave_preserved_text(raw_text: str, spans: list[str]) -> str:
    repaired = fallback_safe_boundary_repair(raw_text, spans)
    if repaired is None:
        return raw_text
    return repaired[0]


def word_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9'’-]{2,}", text)}


def validate_augmented(raw_text: str, augmented_text: str, min_thoughts: int, max_thoughts: int) -> dict[str, Any]:
    think_count = len(re.findall(r"<think>", augmented_text, flags=re.IGNORECASE))
    close_count = len(re.findall(r"</think>", augmented_text, flags=re.IGNORECASE))
    stripped_text = remove_think_spans(augmented_text)
    stripped_words = word_set(stripped_text)
    raw_words = word_set(raw_text)
    coverage = len(raw_words & stripped_words) / max(1, len(raw_words))
    raw_canonical = canonical_original_text(raw_text)
    stripped_canonical = canonical_original_text(stripped_text)
    preserved_exactly = raw_canonical == stripped_canonical
    errors = []
    if think_count < min_thoughts:
        errors.append(f"too_few_thoughts:{think_count}")
    if think_count > max_thoughts:
        errors.append(f"too_many_thoughts:{think_count}")
    if think_count != close_count:
        errors.append(f"unbalanced_tags:{think_count}!={close_count}")
    if coverage < 0.60:
        errors.append(f"low_raw_word_coverage:{coverage:.3f}")
    if not preserved_exactly:
        errors.append("original_text_not_preserved")
    return {
        "think_count": think_count,
        "raw_word_coverage": coverage,
        "original_text_preserved": preserved_exactly,
        "errors": errors,
    }


def call_teacher(
    prompt: str,
    endpoint: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    if requests is None:
        raise RuntimeError("requests is required for teacher generation")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request_timeout = (min(30.0, timeout), timeout)
            resp = requests.post(
                url,
                json=payload,
                timeout=request_timeout,
                headers={"Connection": "close"},
            )
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            return parse_augmented_text(message.get("content"))
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"teacher call failed: {last_err}")


def choose_rows(
    input_jsonl: str | Path,
    train_count: int,
    val_count: int,
    seed: int,
    candidate_multiplier: float,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards}), got {shard_index}")
    rng = random.Random(seed)
    train_candidates = max(train_count, int(train_count * candidate_multiplier))
    val_candidates = max(val_count, int(val_count * candidate_multiplier))
    reservoirs = {"train": [], "val": []}
    targets = {"train": train_candidates, "val": val_candidates}
    seen = {"train": 0, "val": 0}
    for row in jsonl_iter(input_jsonl):
        split = row.get("split")
        if split not in reservoirs:
            continue
        seen[split] += 1
        target = targets[split]
        if target <= 0:
            continue
        reservoir = reservoirs[split]
        if len(reservoir) < target:
            reservoir.append(row)
            continue
        replacement_idx = rng.randrange(seen[split])
        if replacement_idx < target:
            reservoir[replacement_idx] = row
    train = reservoirs["train"]
    val = reservoirs["val"]
    rng.shuffle(train)
    rng.shuffle(val)
    if len(train) < train_count or len(val) < val_count:
        raise RuntimeError(
            f"Not enough rows: requested {train_count=} {val_count=}, got train={seen['train']} val={seen['val']}"
        )
    if num_shards > 1:
        train = train[shard_index::num_shards]
        val = val[shard_index::num_shards]
    return train + val


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_meta(path: str | Path, meta: dict[str, Any]) -> None:
    Path(path).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--train-count", type=int, default=8192)
    parser.add_argument("--val-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5337)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--max-augmented-tokens", type=int, default=768)
    parser.add_argument("--tokenizer-cache", default="")
    parser.add_argument("--min-thoughts", type=int, default=3)
    parser.add_argument("--max-thoughts", type=int, default=6)
    parser.add_argument("--teacher-endpoint", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-temperature", type=float, default=0.6)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-max-tokens", type=int, default=1536)
    parser.add_argument("--teacher-timeout", type=float, default=600.0)
    parser.add_argument("--teacher-retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=float, default=1.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--skip-teacher-errors", action="store_true")
    parser.add_argument("--max-skip-rate", type=float, default=0.10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    tokenizer_cache = args.tokenizer_cache or cfg["data"]["tokenizer_repo_cache"]
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_hf_path(tokenizer_cache),
        local_files_only=True,
        trust_remote_code=True,
    )
    rows = choose_rows(
        args.input_jsonl,
        args.train_count,
        args.val_count,
        args.seed,
        args.candidate_multiplier,
        args.num_shards,
        args.shard_index,
    )

    def prepare_item(idx_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, row = idx_row
        raw_ids = list(row["prefix_ids"]) + list(row["original_suffix_ids"])
        raw_ids = raw_ids[: args.chunk_tokens]
        raw_text = tokenizer.decode(raw_ids, skip_special_tokens=True)
        prompt = PROMPT.format(chunk=raw_text, min_thoughts=args.min_thoughts, max_thoughts=args.max_thoughts)
        return {"idx": idx, "row": row, "raw_ids": raw_ids, "raw_text": raw_text, "prompt": prompt}

    def finish_item(item: dict[str, Any], raw_augmented: str) -> dict[str, Any]:
        idx = item["idx"]
        row = item["row"]
        raw_text = item["raw_text"]
        augmented_text = prune_think_spans(normalize_think_tags(raw_augmented), args.max_thoughts)
        validation = validate_augmented(raw_text, augmented_text, args.min_thoughts, args.max_thoughts)
        unsafe_boundaries = has_unsafe_thought_boundaries(raw_text, augmented_text)
        repaired_preservation = False
        repaired_placement = False
        repair_strategy = "none"
        if validation["errors"] or unsafe_boundaries:
            spans = extract_think_spans(raw_augmented, args.max_thoughts)
            for span_count in range(min(len(spans), args.max_thoughts), args.min_thoughts - 1, -1):
                repair_attempt = alignment_repair_interleaving(raw_text, raw_augmented, span_count)
                if repair_attempt is None:
                    repair_attempt = fallback_safe_boundary_repair(raw_text, spans[:span_count])
                if repair_attempt is None:
                    continue
                repaired, repair_strategy = repair_attempt
                repaired_validation = validate_augmented(raw_text, repaired, args.min_thoughts, args.max_thoughts)
                repaired_unsafe_boundaries = has_unsafe_thought_boundaries(raw_text, repaired)
                if not repaired_validation["errors"] and not repaired_unsafe_boundaries:
                    augmented_text = repaired
                    validation = repaired_validation
                    repaired_preservation = True
                    repaired_placement = unsafe_boundaries
                    unsafe_boundaries = False
                    break
        augmented_ids = tokenizer.encode(augmented_text, add_special_tokens=False)[: args.max_augmented_tokens]
        decoded_augmented = tokenizer.decode(augmented_ids, skip_special_tokens=False)
        decoded_validation = validate_augmented(raw_text, decoded_augmented, args.min_thoughts, args.max_thoughts)
        decoded_unsafe_boundaries = has_unsafe_thought_boundaries(raw_text, decoded_augmented)
        errors = validation["errors"] + [f"decoded_{err}" for err in decoded_validation["errors"]]
        if unsafe_boundaries:
            errors.append("unsafe_thought_boundary")
        if decoded_unsafe_boundaries:
            errors.append("decoded_unsafe_thought_boundary")
        if errors or len(augmented_ids) < 32:
            preview = raw_augmented[:600].replace("\n", " ")
            raise RuntimeError(f"invalid augmentation for row={idx} id={row['id']}: {errors} preview={preview!r}")
        return {
            "id": row["id"],
            "split": row["split"],
            "source_sha": row.get("source_sha", ""),
            "raw_chunk_ids": item["raw_ids"],
            "interleaved_thinking_ids": augmented_ids,
            "raw_text": raw_text,
            "augmented_text": decoded_augmented,
            "teacher_model": args.teacher_model,
            "teacher_backend": "http",
            "prompt_kind": "interleaved_thinking",
            "chunk_tokens": args.chunk_tokens,
            "max_augmented_tokens": args.max_augmented_tokens,
            "think_count": decoded_validation["think_count"],
            "raw_word_coverage": decoded_validation["raw_word_coverage"],
            "original_text_preserved": decoded_validation["original_text_preserved"],
            "preservation_repaired": repaired_preservation,
            "placement_repaired": repaired_placement,
            "repair_strategy": repair_strategy,
        }

    def build_one(idx_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        item = prepare_item(idx_row)
        idx = item["idx"]
        row = item["row"]
        last_error = ""
        for attempt in range(args.teacher_retries + 1):
            raw_augmented = call_teacher(
                item["prompt"],
                endpoint=args.teacher_endpoint,
                model=args.teacher_model,
                temperature=args.teacher_temperature,
                top_p=args.teacher_top_p,
                max_tokens=args.teacher_max_tokens,
                timeout=args.teacher_timeout,
                retries=args.teacher_retries,
            )
            try:
                return finish_item(item, raw_augmented)
            except RuntimeError as exc:
                last_error = str(exc)
            if attempt < args.teacher_retries:
                time.sleep(0.75 * (attempt + 1))
        raise RuntimeError(f"invalid augmentation for row={idx} id={row['id']}: {last_error}")

    output_path = Path(args.output_jsonl)
    meta_path = output_path.with_suffix(".meta.json")
    skipped_path = output_path.with_suffix(".skipped.jsonl")
    if output_path.exists() and not args.resume:
        output_path.unlink()
    if meta_path.exists() and not args.resume:
        meta_path.unlink()
    if skipped_path.exists() and not args.resume:
        skipped_path.unlink()

    skipped_existing = {row["id"] for row in jsonl_iter(skipped_path)} if args.resume and skipped_path.exists() else set()
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        existing = {row["id"]: row for row in jsonl_iter(output_path)}
        rows = [row for row in rows if row["id"] not in existing and row["id"] not in skipped_existing]
        print(
            json.dumps(
                {
                    "resume_existing": len(existing),
                    "resume_skipped": len(skipped_existing),
                    "remaining": len(rows),
                }
            ),
            flush=True,
        )

    def record_skipped(row: dict[str, Any], error: str) -> None:
        append_jsonl(
            skipped_path,
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "source_sha": row.get("source_sha", ""),
                "error": error[:500],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    out: list[dict[str, Any]] = []
    skipped_invalid = 0
    started_at = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(build_one, item): item for item in enumerate(rows)}
        done_count = 0
        last_progress_at = time.time()
        while futures:
            done_futures, _ = wait(futures, timeout=args.progress_timeout, return_when=FIRST_COMPLETED)
            if not done_futures:
                elapsed = max(1e-9, time.time() - started_at)
                progress = {
                    "output_jsonl": args.output_jsonl,
                    "status": "stalled",
                    "built_this_run": len(out),
                    "seen_this_run": done_count,
                    "remaining_this_run": len(futures),
                    "resume_existing": len(existing),
                    "skipped_invalid": skipped_invalid,
                    "rows_per_hour": done_count / elapsed * 3600,
                    "eta_hours_remaining": len(futures) / max(1e-9, done_count / elapsed) / 3600,
                    "last_progress_unix": last_progress_at,
                    "progress_timeout": args.progress_timeout,
                }
                write_meta(output_path, progress)
                print(
                    json.dumps(
                        {
                            "error": "no_progress_timeout",
                            "progress_timeout": args.progress_timeout,
                            "message": "aborting immediately so the launcher can restart from the JSONL checkpoint",
                        }
                    ),
                    flush=True,
                )
                os._exit(124)
            for fut in done_futures:
                done_count += 1
                last_progress_at = time.time()
                idx, src_row = futures.pop(fut)
                try:
                    row = fut.result()
                except RuntimeError as exc:
                    if str(exc).startswith("teacher call failed") and not args.skip_teacher_errors:
                        raise
                    if not args.skip_invalid:
                        raise
                    skipped_invalid += 1
                    record_skipped(src_row, str(exc))
                    print(
                        json.dumps(
                            {
                                "skipped_invalid_row": src_row.get("id"),
                                "split": src_row.get("split"),
                                "error": str(exc)[:500],
                            }
                        ),
                        flush=True,
                    )
                    continue
                out.append(row)
                append_jsonl(output_path, row)
                if done_count % args.checkpoint_every == 0 or not futures:
                    elapsed = max(1e-9, time.time() - started_at)
                    rate = done_count / elapsed
                    progress = {
                        "output_jsonl": args.output_jsonl,
                        "status": "running",
                        "built_this_run": len(out),
                        "seen_this_run": done_count,
                        "remaining_this_run": len(futures),
                        "resume_existing": len(existing),
                        "skipped_invalid": skipped_invalid,
                        "rows_per_hour": rate * 3600,
                        "eta_hours_remaining": len(futures) / max(1e-9, rate) / 3600,
                        "last_progress_unix": last_progress_at,
                    }
                    write_meta(output_path, progress)
                    print(json.dumps(progress), flush=True)

    final_rows = list(existing.values()) + out if existing else out
    n = write_jsonl(args.output_jsonl, final_rows) if existing else len(final_rows)
    seen_this_run = len(out) + skipped_invalid
    skip_rate = skipped_invalid / max(1, seen_this_run)
    if skip_rate > args.max_skip_rate:
        raise RuntimeError(
            f"thinking data skip rate too high: {skipped_invalid}/{seen_this_run}={skip_rate:.3f} "
            f"> max_skip_rate={args.max_skip_rate:.3f}"
        )
    meta = {
        "output_jsonl": args.output_jsonl,
        "rows": n,
        "train_count": sum(1 for row in final_rows if row["split"] == "train"),
        "val_count": sum(1 for row in final_rows if row["split"] == "val"),
        "chunk_tokens": args.chunk_tokens,
        "max_augmented_tokens": args.max_augmented_tokens,
        "teacher_endpoint": args.teacher_endpoint,
        "teacher_model": args.teacher_model,
        "teacher_backend": "http",
        "prompt_kind": "interleaved_thinking",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "avg_think_count": sum(row["think_count"] for row in final_rows) / max(1, len(final_rows)),
        "avg_raw_word_coverage": sum(row["raw_word_coverage"] for row in final_rows) / max(1, len(final_rows)),
        "original_text_preserved_rate": sum(float(row["original_text_preserved"]) for row in final_rows)
        / max(1, len(final_rows)),
        "preservation_repaired_rate": sum(float(row.get("preservation_repaired", False)) for row in final_rows)
        / max(1, len(final_rows)),
        "placement_repaired_rate": sum(float(row.get("placement_repaired", False)) for row in final_rows)
        / max(1, len(final_rows)),
        "repair_strategy_counts": {
            strategy: sum(1 for row in final_rows if row.get("repair_strategy") == strategy)
            for strategy in sorted({row.get("repair_strategy", "none") for row in final_rows})
        },
        "skipped_invalid": skipped_invalid,
        "skip_rate": skip_rate,
        "teacher_timeout": args.teacher_timeout,
        "max_skip_rate": args.max_skip_rate,
        "status": "complete",
    }
    write_meta(args.output_jsonl, meta)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
