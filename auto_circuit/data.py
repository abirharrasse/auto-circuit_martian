import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as t
import torch.utils.data
from attr import dataclass
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

BatchKey = int


@dataclass(frozen=True)
class PromptPair:
    clean: t.Tensor
    corrupt: t.Tensor
    answers: t.Tensor
    wrong_answers: t.Tensor


@dataclass(frozen=True)
class PromptPairBatch:
    key: BatchKey
    diverge_idx: int
    clean: t.Tensor
    corrupt: t.Tensor
    answers: List[t.Tensor] | t.Tensor
    wrong_answers: List[t.Tensor] | t.Tensor


def collate_fn(batch: List[PromptPair]) -> PromptPairBatch:
    clean = t.stack([p.clean for p in batch])
    corrupt = t.stack([p.corrupt for p in batch])
    if all([p.answers.shape == batch[0].answers.shape for p in batch]):
        answers = t.stack([p.answers for p in batch])
    else:  # Sometimes each prompt has a different number of answers
        answers = [p.answers for p in batch]
    if all([p.wrong_answers.shape == batch[0].wrong_answers.shape for p in batch]):
        wrong_answers = t.stack([p.wrong_answers for p in batch])
    else:  # Sometimes each prompt has a different number of wrong answers
        wrong_answers = [p.wrong_answers for p in batch]
    key = hash((str(clean.tolist()), str(corrupt.tolist())))

    diverge_idxs = (~(clean == corrupt)).int().argmax(dim=1)
    diverge_idx: int = int(diverge_idxs.min().item())
    return PromptPairBatch(key, diverge_idx, clean, corrupt, answers, wrong_answers)


class PromptDataset(Dataset):
    def __init__(
        self,
        clean_prompts: List[t.Tensor] | t.Tensor,
        corrupt_prompts: List[t.Tensor] | t.Tensor,
        answers: List[t.Tensor],
        wrong_answers: List[t.Tensor],
    ):
        self.clean_prompts = clean_prompts
        self.corrupt_prompts = corrupt_prompts
        self.answers = answers
        self.wrong_answers = wrong_answers

    def __len__(self) -> int:
        assert len(self.clean_prompts) == len(self.corrupt_prompts)
        return len(self.clean_prompts)

    def __getitem__(self, idx: int) -> PromptPair:
        return PromptPair(
            self.clean_prompts[idx],
            self.corrupt_prompts[idx],
            self.answers[idx],
            self.wrong_answers[idx],
        )


class PromptDataLoader(DataLoader[PromptPairBatch]):
    def __init__(
        self,
        *args: Any,
        seq_len: Optional[int],
        diverge_idx: int,
        seq_labels: Optional[List[str]] = None,
        kv_cache: Optional[HookedTransformerKeyValueCache] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.diverge_idx = diverge_idx
        self.seq_labels = seq_labels
        assert kv_cache is None or diverge_idx > 0
        self.kv_cache = kv_cache


def load_datasets_from_json(
    model: Optional[t.nn.Module],
    path: Path,
    device: t.device,
    prepend_bos: bool = True,
    batch_size: int | Tuple[int, int] = 32,  # (train, test) if tuple
    train_test_split: Sequence[int | float] = [0.9, 0.1],
    length_limit: int = 100,
    return_seq_length: bool = False,
    tail_divergence: bool = False,  # Remove all tokens before divergence
    random_subet: bool = True,
    pad: bool = True,
) -> Tuple[PromptDataLoader, PromptDataLoader]:
    """Load a dataset from a json file. The file should specify a list of
    dictionaries with keys "clean_prompt" and "corrupt_prompt"."""
    with open(path, "r") as f:
        data = json.load(f)
    random.shuffle(data["prompts"]) if random_subet else None
    clean_prompts = [d["clean"] for d in data["prompts"]][:length_limit]
    corrupt_prompts = [d["corrupt"] for d in data["prompts"]][:length_limit]
    answer_strs = [d["answers"] for d in data["prompts"]][:length_limit]
    wrong_answer_strs = [d["wrong_answers"] for d in data["prompts"]][:length_limit]
    seq_labels = data.get("seq_labels", None)
    kvs = []
    if model is None:
        clean_prompts = [t.tensor(p).to(device) for p in clean_prompts]
        corrupt_prompts = [t.tensor(p).to(device) for p in corrupt_prompts]
        answers = [t.tensor(a).to(device) for a in answer_strs]
        wrong_answers = [t.tensor(a).to(device) for a in wrong_answer_strs]
        seq_len = clean_prompts[0].shape[0]
        assert not tail_divergence
        diverge_idx = 0
    else:
        tokenizer: Any = model.tokenizer
        if prepend_bos:
            clean_prompts = [tokenizer.bos_token + prompt for prompt in clean_prompts]
            corrupt_prompts = [
                tokenizer.bos_token + prompt for prompt in corrupt_prompts
            ]
        tokenizer.padding_side = "left"
        clean_prompts = tokenizer(clean_prompts, padding=pad, return_tensors="pt")
        corrupt_prompts = tokenizer(corrupt_prompts, padding=pad, return_tensors="pt")
        seq_len = None
        if return_seq_length:
            assert t.all(clean_prompts["attention_mask"] == 1)
            assert t.all(corrupt_prompts["attention_mask"] == 1)
            seq_len = clean_prompts["input_ids"].shape[1]
        ans_dict: List[Dict] = [tokenizer(a, return_tensors="pt") for a in answer_strs]
        wrong_ans_dict: List[Dict] = [
            tokenizer(a, return_tensors="pt") for a in wrong_answer_strs
        ]
        clean_prompts = clean_prompts["input_ids"].to(device)
        corrupt_prompts = corrupt_prompts["input_ids"].to(device)
        answers = [a["input_ids"].squeeze(-1).to(device) for a in ans_dict]
        wrong_answers = [a["input_ids"].squeeze(-1).to(device) for a in wrong_ans_dict]
        diverge_idxs = (~(clean_prompts == corrupt_prompts)).int().argmax(dim=1)
        diverge_idx: int = int(diverge_idxs.min().item())
        if tail_divergence and diverge_idx > 0:
            seq_labels = seq_labels[diverge_idx:] if seq_labels is not None else None
            clean_prompts = clean_prompts[:, diverge_idx:]
            corrupt_prompts = corrupt_prompts[:, diverge_idx:]
            clean_len, corrupt_len = clean_prompts.shape[0], corrupt_prompts.shape[0]
            prefixs, cfg, device = [], model.cfg, model.cfg.device
            if isinstance(batch_size, tuple):
                prefixs.append(clean_prompts[: (bs0 := batch_size[0]), :diverge_idx])
                prefixs.append(clean_prompts[: (bs1 := batch_size[1]), :diverge_idx])
                assert clean_len % bs0 == 0 and clean_len % bs1 == 0
                assert corrupt_len % bs0 == 0 and corrupt_len % bs1 == 0
                kvs.append(HookedTransformerKeyValueCache.init_cache(cfg, device, bs0))
                kvs.append(HookedTransformerKeyValueCache.init_cache(cfg, device, bs1))
            else:
                assert clean_len % batch_size == 0 and corrupt_len % batch_size == 0
                prefixs.append(clean_prompts[:batch_size, :diverge_idx])
                kvs.append(
                    HookedTransformerKeyValueCache.init_cache(cfg, device, batch_size)
                )

            for prefix, kv_cache in zip(prefixs, kvs):
                with t.inference_mode():
                    model(prefix, past_kv_cache=kv_cache)
                kv_cache.freeze()

            print("seq_len before divergence", seq_len)
            if return_seq_length:
                assert seq_len is not None
                seq_len -= diverge_idx
            print("seq_len after divergence", seq_len)

    dataset = PromptDataset(clean_prompts, corrupt_prompts, answers, wrong_answers)
    train_set, test_set = torch.utils.data.random_split(dataset, train_test_split)
    train_loader = PromptDataLoader(
        train_set,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        seq_labels=seq_labels,
        kv_cache=kvs[0] if len(kvs) > 0 else None,
        batch_size=batch_size[0] if isinstance(batch_size, tuple) else batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=tail_divergence,
    )
    test_loader = PromptDataLoader(
        test_set,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        seq_labels=seq_labels,
        kv_cache=kvs[-1] if len(kvs) > 0 else None,
        batch_size=batch_size[1] if isinstance(batch_size, tuple) else batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=tail_divergence,
    )
    return train_loader, test_loader
