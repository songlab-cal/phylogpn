from typing import List, Callable
from collections import defaultdict

import torch

def batch_sequences(sequences: List[str], chunk_size: int, tokenizer: Callable) -> dict:
    batch = []
    remainder = []
    batch_indices = []
    remainder_indices = []

    for i, seq in enumerate(sequences):
        ids = tokenizer(seq)
        for start_idx in range(0, len(ids), chunk_size):
            stop_idx = min(start_idx + chunk_size, len(ids))
            chunk = ids[start_idx:stop_idx]

            if len(chunk) == chunk_size:
                batch.append(chunk)
                batch_indices.append(i)
            else:
                remainder.append(chunk)
                remainder_indices.append(i)

    return {
        "batch": torch.stack(batch, dim=0),
        "remainder": remainder,
        "batch_indices": batch_indices,
        "remainder_indices": remainder_indices
    }


def concat_embeddings(batch_embeddings: torch.Tensor, remainder_embeddings: List[torch.tensor], batch_indices: List[int], remainder_indices: List[int]) -> torch.Tensor:
    assert len(batch_embeddings) == len(batch_indices), "Batch embeddings and batch indices have different lengths"
    assert len(remainder_embeddings) == len(remainder_indices), "Remainder embeddings and remainder indices have different lengths"
    
    embedding_dict = defaultdict(list)

    for idx, embedding in zip(batch_indices, batch_embeddings):
        embedding_dict[idx].extend(list(embedding))

    for idx, embedding in zip(remainder_indices, remainder_embeddings):
        embedding_dict[idx].extend(list(embedding))

    stacked_embeddings = []

    for idx in range(max(batch_indices + remainder_indices) + 1):
        embeddings = embedding_dict[idx]

        if embeddings:
            stacked_embeddings.append(torch.stack(embeddings, dim=0).unsqueeze(0))

    return stacked_embeddings