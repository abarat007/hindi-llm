"""Hindi chat template.

We use a ChatML-style format built at the *token* level (not by string
formatting), so the special role tokens map to exactly one id each and loss
masking is unambiguous. A conversation is encoded as:

    <bos> <system> {system text} <eos>
          <user>   {user text}   <eos>
          <assistant> {assistant text} <eos>

Each role token (`<system>`/`<user>`/`<assistant>`) opens a turn; `<eos>`
terminates it (it doubles as the turn separator and the end-of-sequence token,
the way ChatML's `<|im_end|>` does). For inference we stop after emitting the
opening `<assistant>` token (the "generation prompt") and let the model
continue.

The human-readable rendering below mirrors the spec's template:

    <system>
    आप एक सहायक, स्पष्ट और ईमानदार हिंदी सहायक हैं।
    </system>
    <user>
    {user_message}
    </user>
    <assistant>
    {assistant_message}
    </assistant>

(The closing `</...>` tags are only for human display; on the wire we use the
opening role token plus `<eos>`, since the closing tags are not vocab items.)
"""

from __future__ import annotations

from dataclasses import dataclass

from .tokenizer_io import HindiTokenizer

DEFAULT_SYSTEM = "आप एक सहायक, स्पष्ट और ईमानदार हिंदी सहायक हैं।"

ROLE_TOKENS = {
    "system": "<system>",
    "user": "<user>",
    "assistant": "<assistant>",
}


@dataclass
class Turn:
    role: str   # "system" | "user" | "assistant"
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLE_TOKENS:
            raise ValueError(f"unknown role {self.role!r}; expected {list(ROLE_TOKENS)}")


def normalize_messages(
    messages: list[Turn] | list[dict], default_system: str | None = DEFAULT_SYSTEM
) -> list[Turn]:
    """Coerce dicts -> Turn and ensure a leading system turn exists."""
    turns: list[Turn] = []
    for m in messages:
        turns.append(m if isinstance(m, Turn) else Turn(m["role"], m["content"]))
    if default_system is not None and (not turns or turns[0].role != "system"):
        turns = [Turn("system", default_system)] + turns
    return turns


def build_chat_ids(
    tok: HindiTokenizer,
    messages: list[Turn] | list[dict],
    add_generation_prompt: bool = False,
    default_system: str | None = DEFAULT_SYSTEM,
) -> tuple[list[int], list[bool]]:
    """Encode a conversation to token ids + an assistant mask.

    Returns ``(ids, assistant_mask)`` where ``assistant_mask[j]`` is True iff
    ``ids[j]`` is an assistant-produced token (its content or its terminating
    `<eos>`). The role marker token itself is False — it's the cue we give the
    model, not something it must predict.

    With ``add_generation_prompt=True`` the sequence ends with a lone
    `<assistant>` token (no content, no eos), ready for generation.
    """
    turns = normalize_messages(messages, default_system)

    ids: list[int] = [tok.bos_id]
    mask: list[bool] = [False]

    for turn in turns:
        role_id = tok.token_to_id(ROLE_TOKENS[turn.role])
        content_ids = tok.encode(turn.content)
        is_assistant = turn.role == "assistant"

        ids.append(role_id)
        mask.append(False)                      # never train on the role marker

        ids.extend(content_ids)
        mask.extend([is_assistant] * len(content_ids))

        ids.append(tok.eos_id)
        mask.append(is_assistant)               # train to emit the closing eos

    if add_generation_prompt:
        ids.append(tok.token_to_id(ROLE_TOKENS["assistant"]))
        mask.append(False)

    return ids, mask


def render_text(
    messages: list[Turn] | list[dict], default_system: str | None = DEFAULT_SYSTEM
) -> str:
    """Human-readable rendering (for previews/docs), with closing tags."""
    turns = normalize_messages(messages, default_system)
    blocks = []
    for t in turns:
        tag = t.role
        blocks.append(f"<{tag}>\n{t.content}\n</{tag}>")
    return "\n".join(blocks)
