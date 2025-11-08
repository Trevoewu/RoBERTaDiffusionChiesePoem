import argparse
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORT", "1")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


MODEL_DIR = "weights/diffusion-style-mordern-chinese-poetry-great-try"
GPT2_MODEL_DIR = "uer/gpt2-chinese-cluecorpussmall"
DATASET_NAME = "l0ulan/chinese_modern_poems"
MAX_LEN = 256
N_STEPS = 10
TOP_K = 50
TOP_P = 0.95
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiment")


mask_probs: List[float] = [i / N_STEPS for i in range(N_STEPS - 1, -1, -1)]


@dataclass
class SampleResult:
    original: str
    masked: str
    completed: str
    accuracy: float


def prepare_text(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return "/".join(lines)


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        kth = torch.topk(filtered, top_k)[0][..., -1]
        filtered[filtered < kth] = float("-inf")
    if top_p > 0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        indices_to_remove = sorted_indices[remove]
        filtered[indices_to_remove] = float("-inf")
    return filtered


def is_separator(tokenizer: AutoTokenizer, token: str) -> bool:
    rendered = tokenizer.convert_tokens_to_string([token]).strip()
    return rendered == "/"


def select_mask_positions(
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_ratio: float,
    rng: random.Random,
) -> List[int]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    candidates: List[int] = []
    specials = set(tokenizer.all_special_ids)
    for idx, token_id in enumerate(input_ids.tolist()):
        if attention_mask[idx] == 0:
            continue
        if token_id in specials:
            continue
        token = tokens[idx]
        if is_separator(tokenizer, token):
            continue
        candidates.append(idx)
    if not candidates:
        return []
    count = round(mask_ratio * len(candidates))
    count = max(0, min(len(candidates), count))
    if count == 0:
        return []
    return rng.sample(candidates, count)


def diffusion_denoise(
    model: AutoModelForMaskedLM,
    device: torch.device,
    initial_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    editable: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    current_ids = initial_ids.clone()
    mask_tensor = torch.tensor(mask_token_id, device=device, dtype=current_ids.dtype)
    for p_mask in mask_probs:
        with torch.no_grad():
            outputs = model(input_ids=current_ids, attention_mask=attention_mask)
            logits = outputs.logits
        sampled = torch.empty_like(current_ids)
        for i in range(current_ids.size(1)):
            logit_vec = logits[0, i, :]
            filtered = top_k_top_p_filtering(logit_vec, TOP_K, TOP_P)
            probs = torch.softmax(filtered, dim=-1)
            sampled[0, i] = torch.multinomial(probs, num_samples=1)
        if p_mask == 0.0:
            current_ids[editable] = sampled[editable]
            break
        rand = torch.rand_like(current_ids, dtype=torch.float, device=device)
        re_mask = (rand < p_mask) & editable
        next_ids = current_ids.clone()
        next_ids[editable] = torch.where(re_mask[editable], mask_tensor, sampled[editable])
        current_ids = next_ids
    return current_ids


def gpt2_fill_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_positions: Sequence[int],
) -> torch.Tensor:
    completed = input_ids.clone().to(device)
    mask_set = set(mask_positions)
    seq_len = int(attention_mask[0].sum().item())
    start_token_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.pad_token_id
    )
    if start_token_id is None:
        raise ValueError("GPT-2 tokenizer must provide at least one of BOS/EOS/PAD tokens for generation.")
    past = None
    last_token = torch.tensor([[start_token_id]], device=device, dtype=completed.dtype)
    logits = None
    for idx in range(seq_len):
        with torch.no_grad():
            outputs = model(input_ids=last_token, past_key_values=past, use_cache=True)
            logits = outputs.logits[0, -1, :]
            past = outputs.past_key_values
        if idx in mask_set:
            filtered = top_k_top_p_filtering(logits, TOP_K, TOP_P)
            probs = torch.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            completed[0, idx] = next_token
        else:
            next_token = completed[0, idx : idx + 1]
        last_token = next_token.unsqueeze(0)
    return completed


def render_poem(
    tokenizer: AutoTokenizer,
    token_ids: Sequence[int],
    mask_positions: Sequence[int] | None = None,
    placeholder: str = "____",
) -> str:
    mask_index = set(mask_positions or [])
    tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
    special_ids = set(tokenizer.all_special_ids)
    rendered_tokens: List[str] = []
    for idx, (tok, tid) in enumerate(zip(tokens, token_ids)):
        if idx in mask_index:
            rendered_tokens.append(placeholder)
            continue
        if tid in special_ids or (tokenizer.pad_token_id is not None and tid == tokenizer.pad_token_id):
            continue
        rendered_tokens.append(tok)
    text = tokenizer.convert_tokens_to_string(rendered_tokens)
    return text.strip()


def log_results(model_label: str, device: torch.device, mask_ratio: float, total_masked: int, accuracy: float) -> str:
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{model_label.lower()}_mask{mask_ratio:.2f}_{timestamp}.txt"
    path = os.path.join(EXPERIMENT_DIR, filename)
    with open(path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Model: {model_label}\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Mask ratio: {mask_ratio:.2f}\n")
        log_file.write(f"Total masked tokens: {total_masked}\n")
        log_file.write(f"Token accuracy: {accuracy * 100:.2f}%\n")
    return path


def run_benchmark(
    mask_ratio: float,
    split: str,
    num_samples: int,
    model_dir: str,
    model_type: str,
    device: torch.device,
    report_samples: int,
    seed: int,
    gpt2_model_dir: str | None = None,
) -> tuple[float, int, List[SampleResult]]:
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    raw_dataset = load_dataset(DATASET_NAME)
    if split in raw_dataset:
        dataset = raw_dataset[split]
    elif split == "validation":
        train_val = raw_dataset["train"].train_test_split(test_size=0.1, seed=seed)
        dataset = train_val["test"]
    else:
        raise ValueError(f"Unknown split '{split}'. Available splits: {list(raw_dataset.keys())} or 'validation'.")
    dataset = dataset.filter(lambda ex: ex["content"].strip() != "")
    dataset = dataset.map(lambda ex: {**ex, "content": prepare_text(ex["content"])})
    if num_samples:
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    if model_type == "diffusion":
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.model_max_length = MAX_LEN
        model = AutoModelForMaskedLM.from_pretrained(model_dir)
    elif model_type == "gpt2":
        if gpt2_model_dir is None:
            raise ValueError("gpt2_model_dir must be provided for GPT-2 benchmarking.")
        tokenizer = AutoTokenizer.from_pretrained(gpt2_model_dir)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
            else:
                raise ValueError("GPT-2 tokenizer needs a pad, eos, or bos token for padding.")
        tokenizer.model_max_length = MAX_LEN
        model = AutoModelForCausalLM.from_pretrained(gpt2_model_dir)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        raise ValueError("model_type must be 'diffusion' or 'gpt2'.")
    model.to(device)
    model.eval()
    rng = random.Random(seed)
    total_correct = 0
    total_masked = 0
    samples: List[SampleResult] = []
    for idx in range(len(dataset)):
        poem = dataset[idx]["content"]
        encoded = tokenizer(
            poem,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        mask_positions = select_mask_positions(
            tokenizer,
            input_ids[0].cpu(),
            attention_mask[0].cpu(),
            mask_ratio,
            rng,
        )
        if not mask_positions:
            if len(samples) < report_samples:
                decoded = tokenizer.decode(
                    input_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                samples.append(
                    SampleResult(
                        original=decoded,
                        masked=decoded,
                        completed=decoded,
                        accuracy=1.0,
                    )
                )
            continue
        if model_type == "diffusion":
            masked_ids = input_ids.clone()
            for pos in mask_positions:
                masked_ids[0, pos] = tokenizer.mask_token_id
            editable = torch.zeros_like(masked_ids, dtype=torch.bool)
            for pos in mask_positions:
                editable[0, pos] = True
            completed = diffusion_denoise(
                model,
                device,
                masked_ids,
                attention_mask,
                editable,
                tokenizer.mask_token_id,
            )
            inference_ids = completed
        else:
            completed = gpt2_fill_tokens(
                model,
                tokenizer,
                device,
                input_ids,
                attention_mask,
                mask_positions,
            )
            inference_ids = completed
        original_ids = input_ids[0, mask_positions]
        predicted_ids = inference_ids[0, mask_positions]
        correct = (predicted_ids == original_ids.to(device)).sum().item()
        total_correct += correct
        total_masked += len(mask_positions)
        if len(samples) < report_samples:
            original_text = tokenizer.decode(
                input_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            masked_text = render_poem(
                tokenizer,
                input_ids[0].detach().cpu().tolist(),
                mask_positions=mask_positions,
            )
            completed_text = tokenizer.decode(
                inference_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            samples.append(
                SampleResult(
                    original=original_text,
                    masked=masked_text,
                    completed=completed_text,
                    accuracy=correct / len(mask_positions),
                )
            )
    accuracy = total_correct / total_masked if total_masked else 1.0
    return accuracy, total_masked, samples


def detect_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark diffusion-style poem completion")
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="Mask ratio in [0,1]")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to evaluate")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of poems to evaluate (0 for all)")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR, help="Model checkpoint directory")
    parser.add_argument("--gpt2-model-dir", type=str, default=GPT2_MODEL_DIR, help="GPT-2 checkpoint directory")
    parser.add_argument(
        "--model-type",
        type=str,
        default="diffusion",
        choices=["diffusion", "gpt2", "both"],
        help="Which model to evaluate",
    )
    parser.add_argument("--report-samples", type=int, default=3, help="Number of qualitative samples to print")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device()
    model_types = [args.model_type] if args.model_type != "both" else ["diffusion", "gpt2"]
    for mtype in model_types:
        accuracy, total_masked, samples = run_benchmark(
            mask_ratio=args.mask_ratio,
            split=args.split,
            num_samples=args.num_samples,
            model_dir=args.model_dir,
            model_type=mtype,
            device=device,
            report_samples=args.report_samples,
            seed=args.seed,
            gpt2_model_dir=args.gpt2_model_dir,
        )
        label = "Diffusion" if mtype == "diffusion" else "GPT-2"
        print("=" * 80)
        print(f"Model: {label}")
        print(f"Device: {device}")
        print(f"Mask ratio: {args.mask_ratio:.2f}")
        print(f"Total masked tokens: {total_masked}")
        print(f"Token accuracy: {accuracy * 100:.2f}%")
        for idx, sample in enumerate(samples, start=1):
            print("-" * 80)
            print(f"Sample {idx} accuracy: {sample.accuracy * 100:.2f}%")
            print("Original :", sample.original)
            print("Masked   :", sample.masked)
            print("Completed:", sample.completed)
        log_path = log_results(label, device, args.mask_ratio, total_masked, accuracy)
        print(f"[LOG] Saved benchmark summary to {log_path}")


if __name__ == "__main__":
    main()
