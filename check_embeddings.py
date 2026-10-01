from pathlib import Path
import torch

MODES = [
    "FULL",
    "SEQ",
    "SEQ_CHEM",
    "SEQ_CHEM_STRUCT",
]


def inspect_embedding_file(file_path):
    print("=" * 80)
    print(file_path)

    data = torch.load(file_path, map_location="cpu")

    print(f"Samples : {len(data['ids'])}")

    print(
        "sequence :",
        tuple(data["sequence_tokens"].shape),
        "sum =",
        float(data["sequence_tokens"].abs().sum())
    )

    print(
        "chemical :",
        tuple(data["chemical_tokens"].shape),
        "sum =",
        float(data["chemical_tokens"].abs().sum())
    )

    print(
        "structure :",
        tuple(data["structure_tokens"].shape),
        "sum =",
        float(data["structure_tokens"].abs().sum())
    )

    print(
        "physchem :",
        tuple(data["physchem"].shape),
        "sum =",
        float(data["physchem"].abs().sum())
    )


def main():

    for mode in MODES:

        if mode == "FULL":
            folder = Path("./embeddings/FULL")
        else:
            folder = Path(f"./embeddings/{mode}")

        file = folder / "train_embeddings.pt"

        if file.exists():
            inspect_embedding_file(file)
        else:
            print("=" * 80)
            print(file)
            print("NOT FOUND")


if __name__ == "__main__":
    main()