from sklearn.model_selection import train_test_split


with open("esd.txt", "r") as f:
    data = f.read().splitlines()


results = []


for text in data:
    path, text = text.split(":")
    results.append(f"data/wav/{path}.wav|{text}")


train, val = train_test_split(results, test_size=0.1, random_state=42)


with open("train.txt", "w") as f:
    f.write("\n".join(train))


with open("val.txt", "w") as f:
    f.write("\n".join(val))