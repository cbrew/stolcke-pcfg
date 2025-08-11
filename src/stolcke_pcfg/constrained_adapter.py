from collections.abc import Callable

from .stolcke_parser import StolckeParser


class ConstrainedDecoderAdapter:
    """Expose allowed token IDs for masking logits during LLM decoding."""
    def __init__(
        self,
        parser: StolckeParser,
        token_id_to_str: Callable[[int], str],
        str_to_token_id: Callable[[str], int | None],
        next_token_filter=None,
    ) -> None:
        self.parser = parser
        self.id2s = token_id_to_str
        self.s2id = str_to_token_id
        self.next_token_filter = next_token_filter

    def allowed_token_ids(self) -> set[int]:
        terms = self.parser.allowed_terminals()
        if self.next_token_filter:
            return self.next_token_filter(terms)
        ids: set[int] = set()
        for t in terms:
            tid = self.s2id(t)
            if tid is not None:
                ids.add(tid)
        return ids

    def step_with_token(self, token_id: int) -> bool:
        return self.parser.step(self.id2s(token_id))
