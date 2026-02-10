import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_results(results_dir="results_graphmi"):
    records = []
    for fn in os.listdir(results_dir):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(results_dir, fn), "r") as f:
                    records.append(json.load(f))
            except Exception as e:
                print(f"Failed to read {fn}: {e}")
    return records

def summarize(records):
    df = pd.DataFrame(records)
    if df.empty:
        print("No results found.")
        return None

    grouped = df.groupby("dataset")[["auc", "ap", "time_s"]].agg(["mean", "std"])
    print("\n=== GraphMI Evaluation Summary ===")
    print(grouped.round(4))

    out_path = "results_graphmi/summary.csv"
    grouped.to_csv(out_path)
    print(f"\nSummary saved to {out_path}")

    return grouped  # <-- IMPORTANT: return for plotting


def plot_summary(summary_df):
    out_dir = "results_graphmi"
    os.makedirs(out_dir, exist_ok=True)

    datasets = summary_df.index.tolist()
    auc_means = summary_df['auc']['mean'].tolist()
    ap_means  = summary_df['ap']['mean'].tolist()

    # --- AUC Plot ---
    plt.figure(figsize=(6, 4))
    plt.bar(datasets, auc_means)
    plt.title("GraphMI AUC by Dataset")
    plt.ylabel("AUC")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "auc_plot.png"))
    plt.close()

    # --- AP Plot ---
    plt.figure(figsize=(6, 4))
    plt.bar(datasets, ap_means)
    plt.title("GraphMI AP by Dataset")
    plt.ylabel("Average Precision (AP)")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ap_plot.png"))
    plt.close()

    print("📊 Plots saved to results_graphmi/")


if __name__ == "__main__":
    results = load_results()
    summary_df = summarize(results)
    if summary_df is not None:
        plot_summary(summary_df)
