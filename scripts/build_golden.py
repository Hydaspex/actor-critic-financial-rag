import pandas as pd

from acfr.golden.builder import build_golden

df = pd.read_csv("data/golden_seed.csv")
build_golden(df).to_csv("data/golden_curated.csv", index=False)
print("wrote data/golden_curated.csv")
