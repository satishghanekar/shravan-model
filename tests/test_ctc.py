"""The CTC model's letters (shravan/ctc.py): labels into its alphabet, and its output back into words."""

from shravan.ctc import WORD, alphabet, decode, encode, text

VOCAB = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, WORD: 4, "क": 5, "म": 6, "ं": 7, "ो": 8}


def test_text_keeps_only_letters_words_joined() -> None:
    assert text("कं कम, (मो).") == "कं|कम|मो"
    assert text("  क   म ") == "क|म"
    # the Vedic anusvara is the plain one
    assert text("कꣳ") == "कं"


def test_alphabet_adds_missing_letters_after_the_pretrained_ones() -> None:
    a = alphabet(VOCAB, ["ॐ कम", "ॐ"])
    assert {k: a[k] for k in VOCAB} == VOCAB
    assert a["ॐ"] == len(VOCAB)
    assert len(a) == len(VOCAB) + 1


def test_encode_and_decode_round_trip() -> None:
    a = alphabet(VOCAB, ["कम मो"])
    ids = encode("कम मो", a)
    assert ids == [5, 6, 4, 6, 8]
    # each letter held for a few steps, blanks between: the CTC way of writing it
    frames = [1, 5, 5, 1, 6, 6, 4, 4, 1, 6, 1, 8, 8, 1]
    assert decode(frames, a, blank=1) == "कम मो"


def test_decode_a_letter_twice_needs_a_blank_between() -> None:
    assert decode([5, 5, 5], VOCAB, blank=1) == "क"
    assert decode([5, 1, 5], VOCAB, blank=1) == "कक"
    assert decode([0, 5, 2, 3], VOCAB, blank=1) == "क"
