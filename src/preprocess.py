import pandas as pd


def preprocess(df):
    """
    Preprocess freight rate dataset.
    Works for training, validation, and December prediction datasets.
    """

    # Create a copy
    df = df.copy()

    # Fill missing weight
    if "weight" in df.columns:
        df["weight"] = df["weight"].fillna(df["weight"].median())

    # Fill missing market_index
    if "market_index" in df.columns:
        df["market_index"] = df["market_index"].fillna(df["market_index"].median())

    # Convert date and extract features
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.weekday

        df.drop(columns=["date"], inplace=True)

    return df


if __name__ == "__main__":
    # Test on training data
    df = pd.read_csv("data/train_test.csv")

    processed = preprocess(df)

    print("=" * 60)
    print("FIRST 5 ROWS")
    print(processed.head())

    print("\n" + "=" * 60)
    print("SHAPE")
    print(processed.shape)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print(processed.isnull().sum())

    print("\nPreprocessing completed successfully!")