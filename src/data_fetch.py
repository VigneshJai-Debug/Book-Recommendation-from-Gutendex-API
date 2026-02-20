import requests
import pandas as pd

all_items = []
url = "https://gutendex.com/books/"

while url and len(all_items) < 10000:
    response = requests.get(url).json()
    all_items.extend(response.get("results", []))
    url = response.get("next")

df = pd.json_normalize(all_items)
df = df.head(10000)
df.to_csv("data/raw/gutendex-raw-dataset.csv", index=False)
print(df.shape)
