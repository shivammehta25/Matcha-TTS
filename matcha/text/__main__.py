from matcha.text import text_to_sequence


if __name__ == "__main__":
    text = "こんにちは"
    sequence = text_to_sequence(text, ["jp_cleaners"])
    print(sequence)