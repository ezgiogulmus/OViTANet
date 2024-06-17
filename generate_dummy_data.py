import os
import pandas as pd
import numpy as np
dummy_gen_df = pd.DataFrame({f"dummy{i}": np.random.randn(100,) for i in range(100)})
dummy_gen_df["case_id"] = np.arange(100)
dummy_df = pd.DataFrame({
    "survival_months": np.random.randint(0, 100, size=100),
    "event": np.random.randint(0, 2, size=100),
    "case_id": np.arange(100),
})
dummy_df.to_csv("./datasets_csv/dummy.csv", index=False)
dummy_gen_df.to_csv("./datasets_csv/dummy_rna.csv.zip", compression="zip", index=False)

dummy_split = pd.DataFrame({"test": np.random.randint(0, 100, 30)})
os.makedirs("./splits/dummy/", exist_ok=True)
dummy_split.to_csv("./splits/dummy/splits_0.csv", index=False)