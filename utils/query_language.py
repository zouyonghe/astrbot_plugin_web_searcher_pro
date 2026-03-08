def _is_kana(char: str) -> bool:
    codepoint = ord(char)
    return 0x3040 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF


def _is_hangul(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _is_han(char: str) -> bool:
    codepoint = ord(char)
    return 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF


def detect_query_language(query: str) -> str:
    if not isinstance(query, str):
        return "auto"

    text = query.strip()
    if not text:
        return "auto"

    if any(_is_kana(char) for char in text):
        return "ja"

    if any(_is_hangul(char) for char in text):
        return "ko"

    if all(ord(char) < 128 for char in text):
        return "en" if any(char.isalpha() for char in text) else "auto"

    han_count = sum(1 for char in text if _is_han(char))
    ascii_word_count = sum(1 for token in text.split() if token.isascii() and any(char.isalpha() for char in token))

    if han_count:
        if ascii_word_count <= 1:
            return "zh"
        return "auto"

    return "auto"
