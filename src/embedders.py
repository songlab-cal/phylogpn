
import torch
import numpy as np
from typing import Any, Iterator, List, Iterable
from functools import partial, lru_cache
import os
from tqdm.auto import tqdm
from batch_utils import *
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseEmbedder():
    """Base class for embedders.
    All embedders should inherit from this class.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the embedder. Calls `load_model` with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments. Passed to `load_model`.
        **kwargs
            Keyword arguments. Passed to `load_model`.
        """
        self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Load the model. Should be implemented by the inheriting class."""
        raise NotImplementedError
    
    def embed(self, sequences:str, *args, **kwargs):
        """Embed a sequence. Should be implemented by the inheriting class.
        
        Parameters
        ----------
        sequences : str
            The sequences to embed.
        """
        raise NotImplementedError

    def __call__(self, sequence: str, *args, **kwargs):
        """Embed a single sequence. Calls `embed` with the given arguments.
        
        Parameters
        ----------
        sequence : str
            The sequence to embed.
        *args
            Positional arguments. Passed to `embed`.
        **kwargs
            Keyword arguments. Passed to `embed`.

        Returns
        -------
        np.ndarray
            The embedding of the sequence.
        """
        return self.embed([sequence], *args, disable_tqdm=True, **kwargs)[0]
    

class PhlyoGPNEmbedder(BaseEmbedder):
    def load_model(self, config_path: str, model_path: str, *args, **kwargs):
        from types import SimpleNamespace
        import yaml
        from phylogpn import RCEByteNet

        self.batch_size = 64

        with open(config_path, "r") as f:
            config = SimpleNamespace(**yaml.safe_load(f))

        # The `involution_indices` args are to specify the indices of complementary bases
        dilation_rates = [config.kernel_size**i for i in range(config.stack_size)] * config.num_stacks
        model_args = {"input_involution_indices": [3, 2, 1, 0, 4, 5],
            "output_involution_indices": [3, 2, 1, 0],
            "dilation_rates": dilation_rates,
            "outer_dim": config.outer_dim,
            "inner_dim": config.inner_dim,
            "kernel_size": config.kernel_size,
            "pad_token_idx": 5
        }

        self.model = RCEByteNet(**model_args)
        checkpoint = torch.load(model_path, weights_only=True, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self._embedding_dim = config.outer_dim

        self._vocab = {k: v for v, k in enumerate("ACGTN-")}
        self._pad_token_idx = 5

    def tokenizer(self, seqs):
        max_length = max(len(x) for x in seqs)
        padded_seqs = [x.ljust(max_length, "-") for x in seqs]
        indices = [self._vocab.get(x, self._pad_token_idx) for x in padded_seqs]
        return torch.tensor(indices, dtype=torch.long)
    
    @property
    @lru_cache(maxsize=1)
    def _context_size(self):
        context_size = 1

        for i, block in enumerate(self.model.blocks):
            context_size += (block.kernel_size - 1) * block.dilation_rate
            
        return context_size
    
    def _get_seq_chunks(self, seq, chunk_size: int = 1, context_size: int = 1):
        seq_length = len(seq)

        for start_idx in range(0, len(seq), chunk_size):
            stop_idx = min(start_idx + chunk_size, seq_length)
            start_idx = start_idx - context_size // 2
            stop_idx = stop_idx + context_size // 2

            left_padding = right_padding = ""

            if start_idx < 0:
                left_padding = "N" * -start_idx
                start_idx = 0

            if stop_idx > seq_length:
                right_padding = "N" * (stop_idx - seq_length)
                stop_idx = seq_length

            yield left_padding + seq[start_idx:stop_idx] + right_padding

    def _batch_iterator(iterable: Iterator[Any], batch_size: int) -> Iterator[List[Any]]:
        iterable = iter(iterable)
        batch = []
        
        for item in iterable:
            batch.append(item)

            if len(batch) == batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch


    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = True):
        # Chunk up sequences and extend the chunks to include context
        seq_chunks = []
        chunk_idx_dict = defaultdict(list)
        chunk_idx = 0
        chunk_size = 2 * self._context_size
        chunk_size = 512

        for seq_idx, seq in enumerate(sequences):
            chunks = list(self._get_seq_chunks(seq, chunk_size, self._context_size))
            seq_chunks.extend(chunks)
            chunk_idx_dict[seq_idx].extend(list(range(chunk_idx, chunk_idx + len(chunks))))
            chunk_idx += len(chunks)

        batched_seq_dict = batch_sequences(seq_chunks, chunk_size=chunk_size + self._context_size - 1, tokenizer=self.tokenizer)
        dataset = torch.utils.data.TensorDataset(batched_seq_dict["batch"])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        batch_chunk_embeddings = []
        remainder_chunk_embeddings = []
        
        embedder = self.model.encode

        with torch.no_grad():
            for input_tensor in tqdm(data_loader, disable=disable_tqdm, total=len(data_loader)):
                input_tensor = input_tensor[0]
                input_tensor = input_tensor.to(device)
                batch_chunk_embeddings.append(embedder(input_tensor))

            for input_tensor in tqdm(batched_seq_dict["remainder"], disable=disable_tqdm, total=len(batched_seq_dict["remainder"])):
                input_tensor = input_tensor.unsqueeze(0).to(device)
                embedding = embedder(input_tensor).squeeze(0)
                remainder_chunk_embeddings.append(embedding)

        batch_chunk_embeddings_tensor = torch.concat(batch_chunk_embeddings, dim=0) if len(batch_chunk_embeddings) > 0 else torch.tensor([])
        stacked_chunk_embeddings = concat_embeddings(batch_chunk_embeddings_tensor, remainder_chunk_embeddings, batched_seq_dict["batch_indices"], batched_seq_dict["remainder_indices"])
        stacked_chunk_embeddings = [x.detach().to("cpu").numpy() for x in stacked_chunk_embeddings]
        
        seq_embeddings = []

        for seq_idx, chunk_indices in chunk_idx_dict.items():
            seq_embedding = np.concatenate([stacked_chunk_embeddings[i] for i in chunk_indices], axis=1)
            seq_embeddings.append(seq_embedding)
 
        return seq_embeddings
    
class PhlyoGPNPooledEmbedder(BaseEmbedder):
    def load_model(self, config_path: str, model_path: str, *args, **kwargs):
        self._phyloGPN_embedder = PhlyoGPNEmbedder(config_path, model_path, *args, **kwargs)
        context_size = self._phyloGPN_embedder._context_size
        embedding_dim = self._phyloGPN_embedder._embedding_dim
        self._extra_context_size = 6001 - context_size + 1

        self._pooling_func = F.avg_pool1d
        self._linear = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        for param in self._linear.parameters():
            param.requires_grad = False

        seed = 42
        current_rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        self._linear.weight.normal_(mean=0.0, std=1.0)
        torch.set_rng_state(current_rng_state)


    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = True):
        raw_embeddings = self._phyloGPN_embedder.embed(sequences, disable_tqdm, remove_special_tokens, upsample_embeddings)
        pad_size = self._extra_context_size // 2
        lengths = [x.shape[1] for x in raw_embeddings]

        tensors = [torch.tensor(x).squeeze(0) for x in raw_embeddings]
        padded_tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        padded_tensors = padded_tensors.swapaxes(1, 2)
        pooled_tensors = self._pooling_func(padded_tensors, kernel_size=self._extra_context_size, stride=1, padding=pad_size)
        pooled_tensors = pooled_tensors.swapaxes(1, 2)

        pooled_embeddings = []

        for i, length in enumerate(lengths):
            x = pooled_tensors[i, :length, :]
            x = self._linear(x)
            pooled_embeddings.append(x.unsqueeze(0).numpy())

        return [x + y for x, y in zip(raw_embeddings, pooled_embeddings)]


class CaduceusEmbedder(BaseEmbedder):
    def load_model(self, model_name='kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16', *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        self.max_length = self.tokenizer.model_max_length
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True)
        model.to(device)
        self.model = model
        self.batch_size = 256
        
    def tokenize_func(self, sequence):
        sequence = [sequence]
        model_input = self.tokenizer(
            sequence, 
            return_tensors='pt', 
            add_special_tokens=False,
            max_length=self.max_length, 
            truncation=True
            )['input_ids']
        return model_input.squeeze(0)
    
    def encode(self, input_ids):
        return self.model.caduceus(input_ids).last_hidden_state.detach().cpu()
    
    def embed_single_sequence(self, s):
        with torch.inference_mode():
            chunks = [s[chunk : chunk + self.max_length] \
                for chunk in  range(0, len(s), self.max_length)] # split into chunks
            embedded_chunks = []
            for n_chunk, chunk in enumerate(chunks):
                input_ids = self.tokenize_func(chunk)
                input_ids = input_ids.unsqueeze(0).to(device)
                output = self.encode(input_ids)
                embedded_chunks.append(output.numpy())                
            embedding = np.concatenate(embedded_chunks, axis=1)
        return embedding
    
    def embed_batch_sequences(self, sequences, disable_tqdm: bool = False):
        embeddings = []
        chunk_size = min(self.max_length, len(sequences[0]))
        with torch.inference_mode():
            batch_info = batch_sequences(
                sequences=sequences, chunk_size=chunk_size, tokenizer=self.tokenize_func)
            batch_inputs, reminder_inputs, batch_indices, remainder_indices = \
                batch_info["batch"], batch_info["remainder"], \
                batch_info["batch_indices"], batch_info["remainder_indices"]    
            if len(sequences[0]) <= self.max_length:
                assert len(remainder_indices) == 0, "sequences should be of the same length!"
            # handle batched chunks
            batch_inputs_dataset = torch.utils.data.TensorDataset(batch_inputs)
            batch_inputs_dataloader = torch.utils.data.DataLoader(
                dataset=batch_inputs_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=8)
            batch_embeddings = []
            for batch_i, data in tqdm(
                enumerate(batch_inputs_dataloader), disable=disable_tqdm, 
                total=batch_inputs.shape[0]//self.batch_size):
                data = data[0].to(device)
                batch_embeddings_i = self.encode(data) 
                batch_embeddings.append(batch_embeddings_i)
            batch_embeddings = torch.concat(batch_embeddings)
            # handle remainder chunks that are not of length chunk_size
            remainder_embeddings = []
            for chunk in tqdm(reminder_inputs, disable=disable_tqdm):
                chunk = chunk.unsqueeze(0).to(device)
                chunk_embedding = self.encode(chunk)              
                remainder_embeddings.append(chunk_embedding.squeeze(0))
            # organize to sequence-level embeddings
            stacked_embeddings = concat_embeddings(
                batch_embeddings=batch_embeddings, remainder_embeddings=remainder_embeddings, 
                batch_indices=batch_indices, remainder_indices=remainder_indices)  
            embeddings = [embedding.numpy() for embedding in stacked_embeddings]
        return embeddings

    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = False, upsample_embeddings: bool = False):
        '''Embeds a list of sequences using the Caduceus model.
        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to False.
            Only provided for compatibility with other embedders. Caduceus embeddings are already the same length as the input sequence.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.
            Only provided for compatibility with other embedders. Caduceus embeddings are already the same length as the input sequence.
        Returns
        -------

        embeddings : List[np.ndarray]
            List of embeddings.
        '''
        seq_lens = [len(seq) for seq in sequences]
        all_sequences_same_length = np.all([seq_len == seq_lens[0] for seq_len in seq_lens])
        if all_sequences_same_length:
            # all sequences same length, embed in batch
            embeddings = self.embed_batch_sequences(sequences, disable_tqdm=disable_tqdm)
        else:
            # sequences of variable length, embed each separately
            embeddings = [self.embed_single_sequence(seq) for seq in tqdm(sequences, disable=disable_tqdm)]
        return embeddings