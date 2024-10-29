from collections import defaultdict
from typing import Callable, Dict, List, Optional
import pandas as pd
import yaml
from torch.utils.data import Dataset

from fasta_utils import ChunkedSequenceReader


class GenomeChunkDataset(Dataset):
    def __init__(
        self,
        genome_dir_path: str,
        index_path: str,
        chunk_size: int,
        context_size: int = 1,
        filter_: Optional[Callable[[str], bool]] = None,
        pad_char: str = "",
    ):
        assert context_size % 2 == 1, "`context_size` must be an odd number"
        genome_reader = ChunkedSequenceReader(genome_dir_path, index_path)
        self._seq_dict = {x: genome_reader[x] for x in genome_reader.index_df["src"].unique() if filter_(x)}

        metadata_dict = defaultdict(list)

        for name, seq in self._seq_dict.items():
            for start_idx in range(0, len(seq), chunk_size):
                stop_idx = start_idx + chunk_size
                start_idx = start_idx - context_size // 2
                stop_idx = stop_idx + context_size // 2
                metadata_dict["src"].append(name)
                metadata_dict["start"].append(start_idx)
                metadata_dict["stop"].append(stop_idx)

        self._metadata_df = pd.DataFrame(metadata_dict)

        self.pad_char = pad_char

    def __len__(self):
        return len(self._metadata_df)

    def __getitem__(self, idx: int):
        row = self._metadata_df.iloc[idx]
        seq = self._seq_dict[row["src"]]
        seq_length = len(seq)
        start_idx = row["start"]
        stop_idx = row["stop"]
        left_padding = ""
        right_padding = ""

        if start_idx < 0:
            left_padding = self.pad_char * -start_idx
            start_idx = 0

        if stop_idx > seq_length:
            right_padding = self.pad_char * (stop_idx - seq_length)
            stop_idx = seq_length

        return left_padding + seq[start_idx:stop_idx] + right_padding
