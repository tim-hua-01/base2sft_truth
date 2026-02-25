import torch
from dataclasses import dataclass
from typing import Self, cast, Any
from torch import Tensor
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from src.datasets import Message, DialogueDataset

@dataclass
class TokenizedDataset:
    """
    Simplified TokenizedDataset using offset_mapping.
    """
    dialogues: list[list[Message]]
    tokens: Tensor  # Shape: [n_dialogues, seq_len]
    attention_mask: Tensor
    detection_mask: Tensor | None
    tokenizer: PreTrainedTokenizerBase
    encodings: Any  # Stores the raw BatchEncoding output

    @property
    def str_tokens(self) -> list[list[str]]:
        """
        Lazy property: decodes tokens only when accessed (e.g. for debugging).
        """
        token_list = self.tokens.tolist()
        return [
            [self.tokenizer.decode([t]) for t in seq if t != self.tokenizer.pad_token_id] 
            for seq in token_list
        ]
    
    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int | slice) -> Self:
        # If integer index, convert to slice to ensure we always return a TokenizedDataset
        # instance (batch of size 1), consistent with type hinting requirements.
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
            
        return self.__class__(
            dialogues=self.dialogues[idx],
            tokens=self.tokens[idx],
            attention_mask=self.attention_mask[idx],
            detection_mask=self.detection_mask[idx] if self.detection_mask is not None else None,
            tokenizer=self.tokenizer,
            # BatchEncoding usually supports slicing, but we check safety
            encodings=self.encodings[idx] if hasattr(self.encodings, '__getitem__') else None
        )

    @staticmethod
    def _merge_consecutive_messages(dialogue: list[Message]) -> list[dict[str, str]]:
        """
        Merges consecutive messages with the same role into a single dict.
        Useful for splitting a single turn into multiple Message objects for granular detection masks
        without triggering the tokenizer to insert intermediate headers.
        """
        if not dialogue:
            return []
            
        merged = []
        current_role = dialogue[0].role
        current_content = dialogue[0].content
        
        for msg in dialogue[1:]:
            if msg.role == current_role:
                current_content += msg.content
            else:
                merged.append({"role": current_role, "content": current_content})
                current_role = msg.role
                current_content = msg.content
        
        merged.append({"role": current_role, "content": current_content})
        return merged

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace for more robust string matching."""
        import re
        # Replace multiple whitespace with single space, strip edges
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _find_content_flexible(full_text: str, content: str, start_pos: int = 0) -> tuple[int, int]:
        """
        Find content in full_text with flexible whitespace matching.
        Returns (start_char, end_char) or (-1, -1) if not found.
        """
        # First try exact match
        idx = full_text.find(content, start_pos)
        if idx != -1:
            return idx, idx + len(content)
        
        # Try with stripped content
        content_stripped = content.strip()
        idx = full_text.find(content_stripped, start_pos)
        if idx != -1:
            return idx, idx + len(content_stripped)
        
        # Try normalized whitespace match
        import re
        # Create a pattern that matches the content with flexible whitespace
        escaped = re.escape(content_stripped)
        # Allow flexible whitespace between words
        pattern = re.sub(r'\\ ', r'\\s+', escaped)
        match = re.search(pattern, full_text[start_pos:])
        if match:
            return start_pos + match.start(), start_pos + match.end()
        
        return -1, -1

    @classmethod
    def from_dataset(
        cls,
        dataset: DialogueDataset,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int | None = None,
    ) -> Self:
        # 1. Format Dialogues
        # We merge consecutive messages (e.g. split reasoning/conclusion) so the tokenizer
        # sees them as one turn and doesn't insert headers in the middle.
        # We also transform Message objects into standard dicts for apply_chat_template.
        formatted_dialogues = []
        for d in dataset.dialogues:
            merged_conversation = cls._merge_consecutive_messages(d)
            formatted = tokenizer.apply_chat_template(
                merged_conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            formatted_dialogues.append(formatted)

        # 2. Tokenize with Offset Mapping (The Key Simplification)
        # offset_mapping returns (char_start, char_end) for every token.
        encodings = tokenizer(
            formatted_dialogues,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True, 
            add_special_tokens=False
        )

        tokens = encodings["input_ids"]
        # Initialize mask as False
        detect_mask = torch.zeros_like(tokens, dtype=torch.bool)
        
        # Check if offset_mapping is available and valid
        offsets = encodings.get("offset_mapping")
        use_offset_mapping = offsets is not None and len(offsets) > 0
        
        if use_offset_mapping:
            # Convert to tensor if it's a list
            if not isinstance(offsets, Tensor):
                offsets = torch.tensor(offsets)

        # 3. Align Messages to Tokens
        for i, dialogue in enumerate(dataset.dialogues):
            full_text = formatted_dialogues[i]
            search_cursor = 0  # Points to the character index where we start searching
            
            for message in dialogue:
                if not message.detect:
                    # Still need to advance the cursor for non-detect messages
                    start_char, end_char = cls._find_content_flexible(full_text, message.content, search_cursor)
                    if end_char > 0:
                        search_cursor = end_char
                    continue
                
                # Find where this message exists in the formatted string.
                start_char, end_char = cls._find_content_flexible(full_text, message.content, search_cursor)
                
                if start_char == -1:
                    # Fallback: try token-based matching
                    print(f"Warning: Could not find message content in dialogue {i} using character search.")
                    print(f"Attempting token-based fallback...")
                    
                    # Token-based fallback: tokenize the message content and find it in the full tokenization
                    content_tokens = tokenizer.encode(message.content, add_special_tokens=False)
                    full_tokens = tokens[i].tolist()
                    
                    # Find subsequence
                    found = False
                    for j in range(len(full_tokens) - len(content_tokens) + 1):
                        if full_tokens[j:j+len(content_tokens)] == content_tokens:
                            detect_mask[i, j:j+len(content_tokens)] = True
                            found = True
                            break
                    
                    if not found:
                        print(f"  Token-based fallback also failed for dialogue {i}")
                        print(f"  Content: '{message.content[:100]}...'")
                    continue
                    
                # Update cursor so the next message is searched for *after* this one
                search_cursor = end_char

                if use_offset_mapping:
                    # Get the start/end character indices for all tokens in this sequence
                    # token_starts: [seq_len], token_ends: [seq_len]
                    token_starts = offsets[i, :, 0]
                    token_ends = offsets[i, :, 1]
                    
                    # Logic: A token belongs to the message if it overlaps with [start_char, end_char)
                    # We check token_ends > token_starts to ignore special tokens (which often have 0,0 offsets)
                    # Changed: Use overlap logic instead of strict containment
                    mask_indices = (
                        (token_ends > start_char) &  # Token ends after message starts
                        (token_starts < end_char) &   # Token starts before message ends
                        (token_ends > token_starts)   # Valid token (not padding/special)
                    )
                    
                    detect_mask[i] = detect_mask[i] | mask_indices
                else:
                    # Fallback: token-based matching when offset_mapping is not available
                    content_tokens = tokenizer.encode(message.content, add_special_tokens=False)
                    full_tokens = tokens[i].tolist()
                    
                    # Find subsequence
                    for j in range(len(full_tokens) - len(content_tokens) + 1):
                        if full_tokens[j:j+len(content_tokens)] == content_tokens:
                            detect_mask[i, j:j+len(content_tokens)] = True
                            break

        return cls(
            dialogues=dataset.dialogues,
            tokens=tokens,
            attention_mask=encodings["attention_mask"],
            detection_mask=detect_mask,
            tokenizer=tokenizer,
            encodings=encodings
        )

    def verify_detection_mask(self, strict: bool = False):  # Default to non-strict
        for i in range(len(self)):
            input_ids = self.tokens[i]
            mask = self.detection_mask[i]
            output_str = self.tokenizer.decode(input_ids[mask], clean_up_tokenization_spaces=False)
    
            expected_parts = []
            for d in self.dialogues[i]:
                if d.detect:
                    expected_parts.append(d.content)
            expected_str = "".join(expected_parts)
            
            # Strip leading/trailing whitespace for comparison
            assert output_str.strip() == expected_str.strip(), \
                f"Mismatch at dialogue {i}:\nGot: '{output_str}'\nExpected: '{expected_str}'"
        
        print("Tokenized dataset masks verified!")

    def display_detection_mask(self, idx: int):
        """Visual verification of the mask."""
        input_ids = self.tokens[idx]
        mask = self.detection_mask[idx]
        
        print(f"\n--- Visualizing Mask for Dialogue {idx} ---")
        print("RED = Masked (Target), WHITE = Ignored\n")
        
        output_str = ""
        for tok_id, m in zip(input_ids, mask):
            if tok_id == self.tokenizer.pad_token_id: 
                continue
            
            # Decode single token
            word = self.tokenizer.decode([tok_id])
            
            # Simple ANSI color code for Red if masked
            if m:
                output_str += f"\033[91m{word}\033[0m" # Red
            else:
                output_str += word
                
        print(output_str)
        print("\n" + "-"*40)