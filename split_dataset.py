"""
Dataset Split

Train        : 80%
Validation   : 10%
Test         : 10%

Stratified Split
Random Seed = 42
"""
import pandas as pd

from sklearn.model_selection import train_test_split

from config import Config


RANDOM_SEED = 42


def main():

    ####################################################
    # Read full dataset
    ####################################################

    df = pd.read_csv(Config.DATASET_CSV)

    print("=" * 60)
    print("Original Dataset")
    print("=" * 60)

    print(df["label"].value_counts())

    ####################################################
    # First split
    # 70% Train
    # 30% Temp
    ####################################################

    train_df, temp_df = train_test_split(

        df,

        test_size=0.20,

        stratify=df["label"],

        random_state=RANDOM_SEED,

        shuffle=True

    )

    ####################################################
    # Second split
    # 15% Validation
    # 15% Test
    ####################################################

    val_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        stratify=temp_df["label"],

        random_state=RANDOM_SEED,

        shuffle=True

    )

    ####################################################
    # Save
    ####################################################

    train_df.to_csv(

        Config.TRAIN_CSV,

        index=False

    )

    val_df.to_csv(

        Config.VAL_CSV,

        index=False

    )

    test_df.to_csv(

        Config.TEST_CSV,

        index=False

    )

    ####################################################
    # Statistics
    ####################################################

    print()

    print("=" * 60)
    print("TRAIN")
    print("=" * 60)
    print(train_df["label"].value_counts())
    print()

    print(f"Train Size      : {len(train_df)} ({100 * len(train_df) / len(df):.1f}%)")

    print(f"Validation Size : {len(val_df)} ({100 * len(val_df) / len(df):.1f}%)")

    print(f"Test Size       : {len(test_df)} ({100 * len(test_df) / len(df):.1f}%)")
    print()

    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print(val_df["label"].value_counts())

    print()

    print("=" * 60)
    print("TEST")
    print("=" * 60)
    print(test_df["label"].value_counts())

    print()

    print("=" * 60)
    print("Files Saved Successfully")
    print("=" * 60)


if __name__ == "__main__":

    main()
