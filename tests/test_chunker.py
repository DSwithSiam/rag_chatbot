from app.chunker import split_text


def test_split_text_creates_chunks() -> None:
    text = "A " * 2000
    chunks = split_text(text=text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_split_text_validates_overlap() -> None:
    try:
        split_text(text="hello", chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
