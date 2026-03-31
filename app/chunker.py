def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = normalized[start:end]

        # Try to avoid splitting in the middle of a sentence when possible.
        if end < length:
            last_boundary = max(chunk.rfind("."), chunk.rfind("?"), chunk.rfind("!"), chunk.rfind("\n"))
            if last_boundary > int(chunk_size * 0.5):
                end = start + last_boundary + 1
                chunk = normalized[start:end]

        chunks.append(chunk.strip())
        if end >= length:
            break
        start = end - chunk_overlap

    return [c for c in chunks if c]
