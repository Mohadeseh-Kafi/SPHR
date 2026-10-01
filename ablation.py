import os
import json


import torch
import pandas as pd

from config import Config
from model import HCMCAModelV2
from train import train
import evaluation

ABLATION_MODES = [
    "SEQ",
    "SEQ_CHEM",
    "SEQ_CHEM_STRUCT",
    "ALL_NO_ATTENTION",
    #"FULL",
]
from pathlib import Path

def get_embedding_prefix(mode):

    p = Path("./embeddings") / mode

    if p.exists():
        return p

    return Path("./embeddings/FULL")

def run_ablation():

    all_results = {}

    ######################################################
    # Data
    ######################################################

    from EmbeddingDataset import EmbeddingDataset

    from dataloader import create_dataloader
    ######################################################
    # Dataset
    ######################################################

    ######################################################
    # Build Dataset
    ######################################################
    ######################################################

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    ######################################################

    for mode in ABLATION_MODES:

        print()

        print("=" * 70)

        print(f"Ablation Mode : {mode}")

        print("=" * 70)
        emb_dir = get_embedding_prefix(mode)

        train_dataset = EmbeddingDataset(
            os.path.join(emb_dir, "train_embeddings.pt")
        )

        val_dataset = EmbeddingDataset(
            os.path.join(emb_dir, "val_embeddings.pt")
        )

        test_dataset = EmbeddingDataset(
            os.path.join(emb_dir, "test_embeddings.pt")
        )

        ######################################################
        # DataLoader
        ######################################################

        train_loader = create_dataloader(

            train_dataset,

            batch_size=Config.BATCH_SIZE,

            shuffle=True,

            num_workers=Config.NUM_WORKERS

        )

        val_loader = create_dataloader(

            val_dataset,

            batch_size=Config.BATCH_SIZE,

            shuffle=False,

            num_workers=Config.NUM_WORKERS

        )

        test_loader = create_dataloader(

            test_dataset,

            batch_size=Config.BATCH_SIZE,

            shuffle=False,

            num_workers=Config.NUM_WORKERS

        )

        ##################################################

        Config.ABLATION_MODE = mode

        # مسیر checkpoint اختصاصی برای این Ablation
        Config.CHECKPOINT_PATH = f"./checkpoints/best_model_{mode}.pt"
        print(Config.CHECKPOINT_PATH)

        ##################################################

        model = HCMCAModelV2().to(device)

        ##################################################
        print(f"START {mode}")
        train(

            model,

            train_loader,

            val_loader

        )
        print(f"FINISH {mode}")
        print("<<< back to run_ablation >>>")
        ##################################################
        model = evaluation.load_best_model(
            model,
            device,
            Config.CHECKPOINT_PATH
        )

        results = evaluation.evaluate_test(

            model,

            test_loader,

            device

        )

        ##################################################

        metrics = evaluation.compute_test_metrics(

            results

        )

        ##################################################

        evaluation.generate_all_figures(

            results,

            metrics,

            output_dir=f"./results/{mode}"

        )

        ##################################################

        all_results[mode] = metrics
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    ######################################################

    return all_results


def main():
    results = run_ablation()

    ##########################################################
    # Add FULL model from saved file
    ##########################################################

    results["FULL"] = load_full_model_metrics(
        "./results/test_metrics.json"
    )
    ######################################################

    table = {}

    for mode, m in results.items():

        table[mode] = {

            "Accuracy": m["accuracy"],

            "Precision": m["precision"],

            "Recall": m["recall"],

            "Specificity": m["specificity"],

            "F1": m["f1"],

            "MCC": m["mcc"],

            "AUROC": m["auc"],

            "AUPRC": m["auprc"]

        }

    ######################################################

    df = pd.DataFrame(table).T

    os.makedirs("./results", exist_ok=True)

    df.to_csv(

        "./results/ablation_results.csv"

    )

    print()

    print(df)

    print()

    print("Ablation table saved.")

    save_ablation_results(

        results,

        "./results"

    )

def load_full_model_metrics(

        metrics_file="./results/test_metrics.json"

):

    """
    Load metrics of the FULL model
    from saved JSON file.
    """

    with open(

            metrics_file,

            "r",

            encoding="utf-8"

    ) as f:

        metrics = json.load(f)

    return metrics


def save_ablation_results(

        all_results,

        output_dir="./results"

):
    """
    Save Ablation Study Results

    Parameters
    ----------
    all_results : dict

        {
            "SEQ": metrics,
            "SEQ_CHEM": metrics,
            ...
        }

    output_dir : str
    """

    os.makedirs(output_dir, exist_ok=True)

    ##########################################################

    rows = []

    ##########################################################

    for model_name, m in all_results.items():

        rows.append({

            "Model": model_name,

            "Accuracy": m["accuracy"],

            "Precision": m["precision"],

            "Recall": m["recall"],

            "Specificity": m["specificity"],

            "F1": m["f1"],

            "MCC": m["mcc"],

            "AUROC": m["auc"],

            "AUPRC": m["auprc"]

        })

    ##########################################################

    df = pd.DataFrame(rows)

    ##########################################################
    # Print
    ##########################################################

    print()

    print("=" * 90)

    print("Ablation Study")

    print("=" * 90)

    print(df.round(4))

    print("=" * 90)

    ##########################################################
    # CSV
    ##########################################################

    csv_path = os.path.join(

        output_dir,

        "ablation_study.csv"

    )

    df.to_csv(

        csv_path,

        index=False

    )

    ##########################################################
    # Excel
    ##########################################################

    excel_path = os.path.join(

        output_dir,

        "ablation_study.xlsx"

    )

    df.to_excel(

        excel_path,

        index=False

    )

    ##########################################################
    # Latex
    ##########################################################

    latex_path = os.path.join(

        output_dir,

        "ablation_study.tex"

    )

    with open(

            latex_path,

            "w",

            encoding="utf-8"

    ) as f:

        f.write(

            df.round(4).to_latex(

                index=False,

                float_format="%.4f"

            )

        )

    ##########################################################
    # JSON
    ##########################################################

    json_path = os.path.join(

        output_dir,

        "ablation_study.json"

    )

    with open(

            json_path,

            "w",

            encoding="utf-8"

    ) as f:

        json.dump(

            rows,

            f,

            indent=4

        )

    ##########################################################

    print()

    print("Saved Files")

    print(csv_path)

    print(excel_path)

    print(latex_path)

    print(json_path)

    ##########################################################

    return df


if __name__ == "__main__":
    main()
