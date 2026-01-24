"""Compatibility utilities for different package versions."""

import torch
from typing import List, Set, Tuple


def get_find_pruneable_heads_and_indices():
    """
    Get find_pruneable_heads_and_indices function with fallback for transformers>=4.40.

    The function was removed from transformers.pytorch_utils in version 4.40.0.
    This provides a compatible implementation for newer versions.
    """
    try:
        from transformers.pytorch_utils import find_pruneable_heads_and_indices
        return find_pruneable_heads_and_indices
    except ImportError:
        def find_pruneable_heads_and_indices(
            heads: List[int],
            n_heads: int,
            head_dim: int,
            already_pruned_heads: Set[int],
        ) -> Tuple[Set[int], torch.LongTensor]:
            """
            Finds the heads and their indices taking already_pruned_heads into account.

            Args:
                heads: List of the indices of heads to prune.
                n_heads: The number of heads in the model.
                head_dim: The dimension of each head.
                already_pruned_heads: A set of already pruned heads.

            Returns:
                A tuple with the remaining heads to prune and the indices of heads to prune
                taking already_pruned_heads into account.
            """
            mask = torch.ones(n_heads, head_dim)
            # Convert to set and remove already pruned heads
            heads_to_prune = set(heads) - already_pruned_heads
            for head in heads_to_prune:
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads_to_prune, index

        return find_pruneable_heads_and_indices
