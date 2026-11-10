from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def load_gahd_raw(path_gahd: str | None = None) -> pd.DataFrame:
    if path_gahd is not None:
        return pd.read_csv(path_gahd)
    dataset = load_dataset("jagoldz/gahd", data_files="gahd.csv")
    return pd.DataFrame(dataset["train"])


def load_gahd(
    path_gahd: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_gahd_raw(path_gahd)
    missing = sorted({"gahd_id", "text", "label", "split"} - set(df.columns))
    if missing:
        raise KeyError(f"GAHD lacks columns {missing}")
    if not df["label"].isin([0, 1]).all():
        raise ValueError("GAHD labels must be binary")
    df = df.loc[df["text"].fillna("").str.strip().ne("")].copy()

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_dev = df[df["split"] == "dev"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    return df_train, df_dev, df_test
