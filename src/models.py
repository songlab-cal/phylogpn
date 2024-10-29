from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from types import SimpleNamespace
from typing import Dict, Generator, List, Iterator, Any, Optional, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

from fasta_utils import ChunkedSequenceReader


class VariantEffectPredictor(ABC):
    @abstractmethod
    def __call__(self, sequences: List[str]):
        pass

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_substitutions(seq: str):
    for idx in range(len(seq)):
        reference_allele = seq[idx]

        for alternate_allele in list("ACGT"):
            if alternate_allele != reference_allele:
                yield idx, reference_allele, alternate_allele, seq[:idx] + alternate_allele + seq[idx + 1:]


complement_dict = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def get_reverse_complement(seq: str) -> str:
    return "".join(complement_dict[base] for base in reversed(seq))


class NucleotideTransformer(VariantEffectPredictor):
    def __init__(self, id_: str):
        alias = id_
        id_to_aliases = {}

        suffixes = ["500m-human-ref", "500m-1000g", "2.5b-1000g", "2.5b-multi-species"]

        for suffix in suffixes:
            id_ = f"InstaDeepAI/nucleotide-transformer-{suffix}"
            aliases = [id_, id_.removeprefix("InstaDeepAI/"), suffix]
            id_to_aliases[id_] = aliases

        alias_found = False

        for id_, aliases in id_to_aliases.items():
            if alias in aliases:
                self._tokenizer = AutoTokenizer.from_pretrained(id_, trust_remote_code=True)
                model = AutoModelForMaskedLM.from_pretrained(id_, trust_remote_code=True)
                self._model = model.to(self.device)
                alias_found = True

        if not alias_found:
            raise ValueError(f"Invalid value for `id_`: {alias}")

    @cached_property
    def token_to_idx(self):
        return self._tokenizer.get_vocab()

    def __call__(self, sequences: List[str]) -> List[pd.DataFrame]:
        sequences.extend([get_reverse_complement(seq) for seq in sequences])

        # Process sequences by replacing 6-mers with ambiguous bases with `<unk>`
        subseq_size = 6
        subseqs = []
        processed_seqs = []

        for seq in sequences:
            subseqs.append([])
            processed_seq = ""

            for i in range(0, len(seq), subseq_size):
                subseq = seq[i:i + subseq_size]

                if "N" in subseq or len(subseq) < subseq_size:
                    subseq = "<unk>"

                processed_seq += subseq

                subseqs[-1].append(subseq)

            processed_seqs.append(processed_seq)

        token_ids = self._tokenizer.batch_encode_plus(
            processed_seqs, return_tensors="pt", padding="max_length", max_length=self._tokenizer.model_max_length
        )["input_ids"]
        token_ids = token_ids.to(self.device)

        attention_mask = token_ids != self._tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self._model(token_ids, attention_mask=attention_mask, encoder_attention_mask=attention_mask)

        logits = outputs["logits"][:, 1:].to("cpu").numpy()

        result_df_list = []

        for i, subseqs_ in enumerate(subseqs):
            result_dict = defaultdict(list)

            for j, reference_subseq in enumerate(subseqs_):
                if reference_subseq == "<unk>":
                    continue

                reference_token_idx = self.token_to_idx[reference_subseq]
                reference_log_likelihood = logits[i, j, reference_token_idx]

                for k, reference_allele, alternate_allele, alternate_subseq in generate_substitutions(reference_subseq):
                    result_dict["idx"].append(j * subseq_size + k)
                    result_dict["ref"].append(reference_allele)
                    result_dict["alt"].append(alternate_allele)
                    alternate_token_idx = self.token_to_idx[alternate_subseq]
                    log_likelihood_ratio = logits[i, k, alternate_token_idx] - reference_log_likelihood
                    result_dict["log_likelihood_ratio"].append(log_likelihood_ratio)

            result_df_list.append(pd.DataFrame(result_dict))

        augmented_df_list = []

        for i in range(len(result_df_list) // 2):
            j = i + len(result_df_list) // 2
            result_df = result_df_list[i].copy()
            reverse_complement_df = result_df_list[j].copy()

            if len(result_df) == 0:
                assert len(reverse_complement_df) == 0
                augmented_df_list.append(result_df)
                continue

            slope = (result_df["idx"].min() - result_df["idx"].max()
                     ) // (reverse_complement_df["idx"].max() - reverse_complement_df["idx"].min())
            intercept = result_df["idx"].min() - slope * reverse_complement_df["idx"].max()
            reverse_complement_df["idx"] = reverse_complement_df["idx"].map(lambda x: slope * x + intercept)
            reverse_complement_df["ref"] = reverse_complement_df["ref"].map(complement_dict)
            reverse_complement_df["alt"] = reverse_complement_df["alt"].map(complement_dict)
            reverse_complement_df.sort_values(["idx", "alt"], inplace=True)
            reverse_complement_df.reset_index(drop=True, inplace=True)
            assert (result_df["idx"] == reverse_complement_df["idx"]).all()
            assert (result_df["ref"] == reverse_complement_df["ref"]).all()
            assert (result_df["alt"] == reverse_complement_df["alt"]).all()

            result_df["log_likelihood_ratio"] = (result_df["log_likelihood_ratio"] + reverse_complement_df["log_likelihood_ratio"]) / 2

            augmented_df_list.append(result_df)

        return augmented_df_list


class NucleotideTransformerV2(VariantEffectPredictor):
    def __init__(self, id_: str):
        alias = id_
        id_to_aliases = {}

        for size in [50, 100, 250, 500]:
            id_ = f"InstaDeepAI/nucleotide-transformer-v2-{size}m-multi-species"
            aliases = [
                id_,
                id_.removeprefix("InstaDeepAI/"),
                id_.removeprefix("InstaDeepAI/nucleotide-transformer-v2-"),
            ]
            aliases.append(id_.removeprefix("InstaDeepAI/").removesuffix("-multi-species"))
            aliases.append(id_.removeprefix("InstaDeepAI/nucleotide-transformer-v2-").removesuffix("-multi-species"))
            id_to_aliases[id_] = aliases

        alias_found = False

        for id_, aliases in id_to_aliases.items():
            if alias in aliases:
                self._tokenizer = AutoTokenizer.from_pretrained(id_, trust_remote_code=True)
                model = AutoModelForMaskedLM.from_pretrained(id_, trust_remote_code=True)
                self._model = model.to(self.device)
                alias_found = True

        if not alias_found:
            raise ValueError(f"Invalid value for `id_`: {alias}")

    @cached_property
    def token_to_idx(self):
        return self._tokenizer.get_vocab()

    def __call__(self, sequences: List[str]) -> List[pd.DataFrame]:
        sequences.extend([get_reverse_complement(seq) for seq in sequences])

        # Process sequences by replacing 6-mers with ambiguous bases with `<unk>`
        subseq_size = 6
        subseqs = []
        processed_seqs = []

        for seq in sequences:
            subseqs.append([])
            processed_seq = ""

            for i in range(0, len(seq), subseq_size):
                subseq = seq[i:i + subseq_size]

                if "N" in subseq or len(subseq) < subseq_size:
                    subseq = "<unk>"

                processed_seq += subseq

                subseqs[-1].append(subseq)

            processed_seqs.append(processed_seq)

        token_ids = self._tokenizer.batch_encode_plus(
            processed_seqs, return_tensors="pt", padding="max_length", max_length=self._tokenizer.model_max_length
        )["input_ids"]
        token_ids = token_ids.to(self.device)

        attention_mask = token_ids != self._tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self._model(token_ids, attention_mask=attention_mask, encoder_attention_mask=attention_mask)

        logits = outputs["logits"][:, 1:].to("cpu").numpy()

        result_df_list = []

        for i, subseqs_ in enumerate(subseqs):
            result_dict = defaultdict(list)

            for j, reference_subseq in enumerate(subseqs_):
                if reference_subseq == "<unk>":
                    continue

                reference_token_idx = self.token_to_idx[reference_subseq]
                reference_log_likelihood = logits[i, j, reference_token_idx]

                for k, reference_allele, alternate_allele, alternate_subseq in generate_substitutions(reference_subseq):
                    result_dict["idx"].append(j * subseq_size + k)
                    result_dict["ref"].append(reference_allele)
                    result_dict["alt"].append(alternate_allele)
                    alternate_token_idx = self.token_to_idx[alternate_subseq]
                    log_likelihood_ratio = logits[i, k, alternate_token_idx] - reference_log_likelihood
                    result_dict["log_likelihood_ratio"].append(log_likelihood_ratio)

            result_df_list.append(pd.DataFrame(result_dict))

        augmented_df_list = []

        for i in range(len(result_df_list) // 2):
            j = i + len(result_df_list) // 2
            result_df = result_df_list[i].copy()
            reverse_complement_df = result_df_list[j].copy()

            if len(result_df) == 0:
                assert len(reverse_complement_df) == 0
                augmented_df_list.append(pd.DataFrame())
                continue

            slope = (result_df["idx"].min() - result_df["idx"].max()
                     ) // (reverse_complement_df["idx"].max() - reverse_complement_df["idx"].min())
            intercept = result_df["idx"].min() - slope * reverse_complement_df["idx"].max()
            reverse_complement_df["idx"] = reverse_complement_df["idx"].map(lambda x: slope * x + intercept)
            reverse_complement_df["ref"] = reverse_complement_df["ref"].map(complement_dict)
            reverse_complement_df["alt"] = reverse_complement_df["alt"].map(complement_dict)
            reverse_complement_df.sort_values(["idx", "alt"], inplace=True)
            reverse_complement_df.reset_index(drop=True, inplace=True)
            assert (result_df["idx"] == reverse_complement_df["idx"]).all()
            assert (result_df["ref"] == reverse_complement_df["ref"]).all()
            assert (result_df["alt"] == reverse_complement_df["alt"]).all()

            result_df["log_likelihood_ratio"] = (result_df["log_likelihood_ratio"] + reverse_complement_df["log_likelihood_ratio"]) / 2
            augmented_df_list.append(result_df)

        return augmented_df_list


class PhyloGPN(VariantEffectPredictor):
    def __init__(self):
        from phylogpn import RCEByteNet

        with open(f"PhyloGPN/config.yaml", "r") as f:
            config = SimpleNamespace(**yaml.safe_load(f))

        # The `involution_indices` args are to specify the indices of complementary bases
        dilation_rates = [config.kernel_size**i for i in range(config.stack_size)] * config.num_stacks

        model_args = {
            "input_involution_indices": [3, 2, 1, 0, 4, 5],
            "output_involution_indices": [3, 2, 1, 0],
            "dilation_rates": dilation_rates,
            "outer_dim": config.outer_dim,
            "inner_dim": config.inner_dim,
            "kernel_size": config.kernel_size,
            "pad_token_idx": 5,
        }

        model = RCEByteNet(**model_args).to(self.device)
        checkpoint = torch.load(config.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self._model = model

    @cached_property
    def token_to_idx(self):
        return {x: i for i, x in enumerate("ACGTN")}

    @property
    @lru_cache(maxsize=1)
    def context_size(self):
        context_size = 1

        for block in self._model.blocks:
            context_size += (block.kernel_size - 1) * block.dilation_rate

        return context_size

    def __call__(self, sequences: List[str]):
        seq_indices = []

        for seq in sequences:
            seq_indices.append(torch.tensor([self.token_to_idx.get(x, 4) for x in seq], device=self.device))

        input_tensor = pad_sequence(seq_indices, padding_value=4, batch_first=True)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad(), torch.autocast(device_str):
            outputs = self._model(input_tensor).to("cpu").numpy()

        result_df_list = []

        for i, reference_seq in enumerate(sequences):
            result_dict = defaultdict(list)
            slice_ = slice(self.context_size // 2, -(self.context_size // 2))

            for j, reference_allele, alternate_allele, __ in generate_substitutions(reference_seq[slice_]):
                if reference_allele == "N" or j >= len(reference_seq) - self.context_size // 2:
                    continue

                result_dict["idx"].append(j + self.context_size // 2)
                result_dict["ref"].append(reference_allele)
                result_dict["alt"].append(alternate_allele)
                log_likelihood_ratio = outputs[i, j, self.token_to_idx[alternate_allele]]
                log_likelihood_ratio -= outputs[i, j, self.token_to_idx[reference_allele]]
                result_dict["log_likelihood_ratio"].append(log_likelihood_ratio)

            result_df_list.append(pd.DataFrame(result_dict))

        return result_df_list
    

class HyenaDNA(VariantEffectPredictor):
    def __init__(self, id_: str, chunk_size: int):
        assert chunk_size % 2 == 0, "`chunk_size` must be even"
        self.chunk_size = chunk_size

        alias = id_
        id_to_aliases = {}

        model_list = [
            "LongSafari/hyenadna-large-1m-seqlen-hf",
            "LongSafari/hyenadna-medium-450k-seqlen-hf",
            "LongSafari/hyenadna-medium-160k-seqlen-hf",
            "LongSafari/hyenadna-small-32k-seqlen-hf",
            "LongSafari/hyenadna-tiny-1k-seqlen-hf",
        ]

        for full_id in model_list:
            aliases = [
                full_id,
                full_id.removeprefix("LongSafari/"),
                full_id.removeprefix("LongSafari/hyenadna-"),
                full_id.removeprefix("LongSafari/").removesuffix("-hf"),
                full_id.removeprefix("LongSafari/hyenadna-").removesuffix("-hf"),
                full_id.removeprefix("LongSafari/").removesuffix("-seqlen-hf"),
                full_id.removeprefix("LongSafari/hyenadna-").removesuffix("-seqlen-hf"),
            ]
            id_to_aliases[full_id] = aliases

        alias_found = False

        for full_id, aliases in id_to_aliases.items():
            if alias in aliases:
                self._tokenizer = AutoTokenizer.from_pretrained(full_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(full_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
                self._model = model.to(self.device)
                alias_found = True

        if not alias_found:
            raise ValueError(f"Invalid value for `id_`: {alias}")

    @cached_property
    def token_to_idx(self):
        return self._tokenizer.get_vocab()
    
    @cached_property
    def idx_to_token(self):
        return {v: k for k, v in self.token_to_idx.items()}

    def __call__(self, sequences: List[str]) -> List[pd.DataFrame]:
        sequences.extend([get_reverse_complement(seq) for seq in sequences])
        sequences = [seq[:(len(seq) + 1) // 2 + self.chunk_size // 2] for seq in sequences]
        
        tokenizer_results = self._tokenizer(["[BOS]" + seq for seq in sequences], return_tensors="pt", padding=True)
        token_ids = tokenizer_results["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self._model(token_ids)

        logits = outputs["logits"].to(torch.float16).to("cpu").numpy()
        logits = logits[:, :-2, :] # Shift logits

        result_df_list = []

        for i, reference_seq in enumerate(sequences):
            chunk_start_idx = len(reference_seq) - self.chunk_size

            result_dict = defaultdict(list)

            for j, reference_allele, alternate_allele, __ in generate_substitutions(reference_seq[chunk_start_idx:]):
                j += chunk_start_idx

                if reference_allele == "N":
                    continue

                result_dict["idx"].append(j)
                result_dict["ref"].append(reference_allele)
                result_dict["alt"].append(alternate_allele)
                log_likelihood_ratio = logits[i, j, self.token_to_idx[alternate_allele]]
                log_likelihood_ratio -= logits[i, j, self.token_to_idx[reference_allele]]
                result_dict["log_likelihood_ratio"].append(log_likelihood_ratio)

            result_df_list.append(pd.DataFrame(result_dict))

        augmented_df_list = []

        for i in range(len(result_df_list) // 2):
            j = i + len(result_df_list) // 2
            result_df = result_df_list[i].copy()
            reverse_complement_df = result_df_list[j].copy()

            if len(result_df) == 0:
                assert len(reverse_complement_df) == 0
                augmented_df_list.append(pd.DataFrame())
                continue

            slope = (result_df["idx"].min() - result_df["idx"].max()
                     ) // (reverse_complement_df["idx"].max() - reverse_complement_df["idx"].min())
            intercept = result_df["idx"].min() - slope * reverse_complement_df["idx"].max()
            reverse_complement_df["idx"] = reverse_complement_df["idx"].map(lambda x: slope * x + intercept)
            reverse_complement_df["ref"] = reverse_complement_df["ref"].map(complement_dict)
            reverse_complement_df["alt"] = reverse_complement_df["alt"].map(complement_dict)
            reverse_complement_df.sort_values(["idx", "alt"], inplace=True)
            reverse_complement_df.reset_index(drop=True, inplace=True)
            assert (result_df["idx"] == reverse_complement_df["idx"]).all()
            assert (result_df["ref"] == reverse_complement_df["ref"]).all()
            assert (result_df["alt"] == reverse_complement_df["alt"]).all()

            result_df["log_likelihood_ratio"] = (result_df["log_likelihood_ratio"] + reverse_complement_df["log_likelihood_ratio"]) / 2
            augmented_df_list.append(result_df)

        return augmented_df_list


class Caduceus(VariantEffectPredictor):
    def __init__(self, id_: str, chunk_size: Optional[int] = None):
        self.chunk_size = chunk_size

        alias = id_
        id_to_aliases = {}

        model_list = [
            "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
            "kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
            "kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
        ]

        for full_id in model_list:
            base_name = full_id.split("/")[1]

            # Start building aliases
            aliases = [full_id, base_name]

            # Generate additional aliases based on model characteristics
            if "seqlen-131k" in base_name:
                aliases.extend([
                    "131k",
                    "caduceus-131k",
                    "caduceus_ps_131k",
                    "ps_131k",
                    "caduceus-ps_seqlen-131k",
                ])
            elif "seqlen-1k" in base_name:
                if "d_model-256" in base_name:
                    aliases.extend([
                        "1k_d256",
                        "caduceus-1k-d256",
                        "caduceus_ps_1k_d256",
                        "ps_1k_d256",
                        "caduceus-ps_seqlen-1k_d256",
                    ])
                elif "d_model-118" in base_name:
                    aliases.extend([
                        "1k_d118",
                        "caduceus-1k-d118",
                        "caduceus_ps_1k_d118",
                        "ps_1k_d118",
                        "caduceus-ps_seqlen-1k_d118",
                    ])

            # Map the full model ID to its aliases
            id_to_aliases[full_id] = aliases

        alias_found = False

        # Search for the provided alias in the aliases list
        for full_id, aliases in id_to_aliases.items():
            if alias in aliases:
                self._tokenizer = AutoTokenizer.from_pretrained(full_id, trust_remote_code=True)
                model = AutoModelForMaskedLM.from_pretrained(full_id, trust_remote_code=True)
                self._model = model.to(self.device)
                alias_found = True
                break # Model found, exit the loop

        if not alias_found:
            raise ValueError(f"Invalid value for `id_`: {alias}")

    @cached_property
    def token_to_idx(self):
        return self._tokenizer.get_vocab()
    
    @cached_property
    def idx_to_token(self):
        return {v: k for k, v in self.token_to_idx.items()}

    def __call__(self, sequences: List[str]) -> List[pd.DataFrame]:
        input_ids = self._tokenizer(sequences)["input_ids"]
        input_tensor = torch.tensor(input_ids, device=self.device, dtype=torch.long)

        with torch.no_grad():
            outputs = self._model(input_tensor)

        logits = outputs["logits"].to("cpu").numpy()[:, :-1]

        result_df_list = []

        for i, reference_seq in enumerate(sequences):
            chunk_size = self.chunk_size or len(reference_seq)
            chunk_start_idx = (len(reference_seq) - self.chunk_size) // 2
            chunk_stop_idx = chunk_start_idx + chunk_size + 1

            result_dict = defaultdict(list)

            for j, reference_allele, alternate_allele, __ in generate_substitutions(reference_seq[chunk_start_idx:chunk_stop_idx]):
                j += chunk_start_idx

                if reference_allele == "N":
                    continue

                result_dict["idx"].append(j)
                result_dict["ref"].append(reference_allele)
                result_dict["alt"].append(alternate_allele)
                log_likelihood_ratio = logits[i, j, self.token_to_idx[alternate_allele]]
                log_likelihood_ratio -= logits[i, j, self.token_to_idx[reference_allele]]
                result_dict["log_likelihood_ratio"].append(log_likelihood_ratio)

            result_df_list.append(pd.DataFrame(result_dict))

        return result_df_list