"""
subtrajectory_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class SubtrajectoryTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 30, min_action: int = 0, max_action: int = 30
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action
       
        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: int) -> Union[str, List[str]]:

        subtraj = np.clip(action, a_min=int(self.min_action), a_max=int(self.max_action))
        subtraj_id=self.action_token_begin_idx +subtraj 

        # Handle single element vs. batch
        if subtraj.ndim ==0:
            return self.tokenizer.decode([int(subtraj_id)])
        else:
            return self.tokenizer.batch_decode((subtraj_id).astype(int).tolist())

    def decode_token_ids_to_actions(self, subj_token_ids: int) -> int:
        """
        Returns discrete subtrajectory ID for discrete subtrajectory token IDs.

        """
        subtraj_id = subj_token_ids-self.action_token_begin_idx 
        

        return subtraj_id

    @property
    def vocab_size(self) -> int:
        return self.n_bins
