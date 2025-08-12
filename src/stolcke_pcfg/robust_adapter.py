#!/usr/bin/env python3
"""
Robust constrained decoder adapter that handles multi-token terminals gracefully.

This adapter solves the tokenization brittleness problem by:
1. Tracking partial matches for multi-token terminals
2. Allowing tokens that could start or continue valid terminal sequences  
3. Maintaining a buffer of partial terminal matches
4. Only advancing the parser when complete terminals are matched
"""
from collections.abc import Callable
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

from .stolcke_parser import StolckeParser


@dataclass
class PartialMatch:
    """Represents a partial match of a terminal sequence."""
    terminal: str           # The full terminal from grammar (e.g., '"Alice"')
    token_sequence: List[int]  # Token IDs seen so far  
    remaining_tokens: List[int]  # Token IDs still needed
    
    @property
    def is_complete(self) -> bool:
        return len(self.remaining_tokens) == 0
    
    @property
    def next_expected_token(self) -> Optional[int]:
        return self.remaining_tokens[0] if self.remaining_tokens else None


class RobustConstrainedAdapter:
    """
    Tokenization-resilient constrained decoder adapter.
    
    Handles multi-token terminals by maintaining partial match state and only
    advancing the parser when complete terminal sequences are consumed.
    """
    
    def __init__(
        self,
        parser: StolckeParser,
        token_id_to_str: Callable[[int], str],
        str_to_token_id: Callable[[str], int | None],
        tokenizer,  # Full tokenizer for sequence analysis
    ):
        self.parser = parser
        self.id2str = token_id_to_str
        self.str2id = str_to_token_id
        self.tokenizer = tokenizer
        
        # Partial match tracking
        self.partial_matches: List[PartialMatch] = []
        
        # Cache tokenizations to avoid repeated encoding
        self._terminal_tokenizations: Dict[str, List[int]] = {}
        
    def _get_token_sequence(self, terminal: str) -> List[int]:
        """Get token sequence for a terminal, with caching."""
        if terminal not in self._terminal_tokenizations:
            token_ids = self.tokenizer.encode(terminal, add_special_tokens=False)
            self._terminal_tokenizations[terminal] = token_ids
        return self._terminal_tokenizations[terminal]
    
    def _initialize_partial_matches(self) -> None:
        """Initialize partial matches for all allowed terminals."""
        allowed_terminals = self.parser.allowed_terminals()
        self.partial_matches = []
        
        for terminal in allowed_terminals:
            token_sequence = self._get_token_sequence(terminal)
            if token_sequence:  # Only if tokenization succeeded
                match = PartialMatch(
                    terminal=terminal,
                    token_sequence=[],
                    remaining_tokens=token_sequence.copy()
                )
                self.partial_matches.append(match)
    
    def allowed_token_ids(self) -> Set[int]:
        """Get all token IDs that could continue any partial match."""
        # Initialize partial matches only if we don't have any active ones
        if not self.partial_matches:
            self._initialize_partial_matches()
        
        allowed_tokens: Set[int] = set()
        
        # Collect all tokens that could start or continue partial matches
        for match in self.partial_matches:
            next_token = match.next_expected_token
            if next_token is not None:
                allowed_tokens.add(next_token)
        
        return allowed_tokens
    
    def step_with_token(self, token_id: int) -> bool:
        """
        Attempt to advance with a token.
        
        Returns True if the token was accepted (either as part of ongoing
        partial match or completing a terminal). Returns False if token
        is not valid in current state.
        """
        if not self.partial_matches:
            self._initialize_partial_matches()
        
        # Find partial matches that expect this token
        continuing_matches = []
        
        for match in self.partial_matches:
            if match.next_expected_token == token_id:
                # Create updated match
                new_match = PartialMatch(
                    terminal=match.terminal,
                    token_sequence=match.token_sequence + [token_id],
                    remaining_tokens=match.remaining_tokens[1:]
                )
                continuing_matches.append(new_match)
        
        if not continuing_matches:
            # Token doesn't continue any partial match
            return False
        
        # Update partial matches to only those that can continue
        self.partial_matches = continuing_matches
        
        # Check if any matches are complete
        completed_matches = [m for m in self.partial_matches if m.is_complete]
        
        if completed_matches:
            # Complete a terminal! Advance parser with the terminal string
            # (We can use any completed match since they should be equivalent)
            completed_terminal = completed_matches[0].terminal
            success = self.parser.step(completed_terminal)
            
            if success:
                # Reset partial matches for next terminal
                self.partial_matches = []
                return True
            else:
                # Parser rejected the terminal (shouldn't happen if logic is correct)
                return False
        
        # Token accepted as part of ongoing partial match
        return True
    
    def allowed_token_mask(self, vocab_size: int) -> List[bool]:
        """Return boolean mask for allowed tokens."""
        allowed_ids = self.allowed_token_ids()
        mask = [False] * vocab_size
        for token_id in allowed_ids:
            if 0 <= token_id < vocab_size:
                mask[token_id] = True
        return mask
    
    def get_current_state_info(self) -> Dict:
        """Get debugging info about current state."""
        return {
            'partial_matches': len(self.partial_matches),
            'allowed_terminals': list(self.parser.allowed_terminals()),
            'parser_position': self.parser.pos,
            'parser_accepted': self.parser.accepted(),
            'active_matches': [
                {
                    'terminal': m.terminal,
                    'progress': f"{len(m.token_sequence)}/{len(m.token_sequence) + len(m.remaining_tokens)}",
                    'next_token': m.next_expected_token
                }
                for m in self.partial_matches
            ]
        }


class MultiTokenAwareFilter:
    """
    Enhanced token filter that handles multi-token terminals intelligently.
    
    Instead of only allowing first tokens, this maintains state about
    partial terminal matches and allows appropriate continuation tokens.
    """
    
    def __init__(self, tokenizer, adapter: RobustConstrainedAdapter):
        self.tokenizer = tokenizer
        self.adapter = adapter
    
    def __call__(self, terminals: Set[str]) -> Set[int]:
        """
        Get allowed token IDs for a set of terminal strings.
        
        This is called by the adapter to determine which tokens are valid.
        We delegate to the adapter's logic for consistency.
        """
        return self.adapter.allowed_token_ids()