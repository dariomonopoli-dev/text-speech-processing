from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def load_gahd_raw():
    dataset = load_dataset("jagoldz/gahd", data_files="gahd.csv")
    return pd.DataFrame(dataset["train"])


def load_gahd():

    df = load_gahd_raw()

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_dev   = df[df["split"] == "dev"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)

    return df_train, df_dev, df_test
