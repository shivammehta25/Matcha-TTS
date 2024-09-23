from pyopenjtalk import extract_fullcontext
import re


def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def g2p(text: str, drop_unvoiced_vowels: bool=True) -> list[str]:
    results = []

    labels = extract_fullcontext(text)
    N = len(labels)

    for i, label in enumerate(labels):
        p3 = re.search(r"\-(.*?)\+", label).group(1)

        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        if p3 == "sil":
            assert i == 0 or i == N - 1
            if i == 0:
                results.append("^")
            elif i == N - 1:
                e3 = numeric_feature_by_regex(r"!(\d+)_", label)
                if e3 == 0:
                    results.append("$")
                elif e3 == 1:
                    results.append("?")
            continue
        elif p3 == "pau":
            results.append("_")
            continue
        else:
            results.append(p3)
        
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", label)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", label)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", label)
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", label)

        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[i + 1])

        if a3 == 1 and a2_next == 1:
            results.append("#")
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            results.append("]")
        elif a2 == 1 and a2_next == 2:
            results.append("[")
    
    return results


if __name__ == "__main__":
    text = "こんにちは"
    print(g2p(text))