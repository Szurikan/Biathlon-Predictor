import os
import matplotlib
matplotlib.use("Agg")  # Užtikrina, kad grafikai nebūtų rodomi GUI režimu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_metric_plots(dates, stats_list, output_dir, prefix=""):
    plt.figure(figsize=(12, 10))
    for i, (metric, label) in enumerate(zip(['accuracy', 'precision_1', 'recall_1', 'f1_1'],
                                            ['Accuracy', 'Precision', 'Recall', 'F1-score'])):
        plt.subplot(2, 2, i+1)
        plt.plot(dates, [s[metric] for s in stats_list], 'o-')
        plt.title(f"{label} pagal etapą")
        plt.xlabel("Data")
        plt.ylabel(label)
        plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{prefix}_metrics_over_time.png" if prefix else "metrics_over_time.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def save_accuracy_plot(dates, stats_list, output_dir, prefix=""):
    plt.figure(figsize=(8, 5))
    plt.plot(dates, [s['accuracy'] for s in stats_list], 'o-')
    plt.title("Bendras tikslumas pagal laiką")
    plt.xlabel("Data")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{prefix}_accuracy_over_time.png" if prefix else "accuracy_over_time.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def save_confusion_matrix(cm, output_dir, prefix=""):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ne', 'Taip'], yticklabels=['Ne', 'Taip'])
    plt.title("Bendra sujaukimo matrica (visi etapai)")
    plt.xlabel("Prognozuota klasė")
    plt.ylabel("Tikroji klasė")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{prefix}_confusion_matrix.png" if prefix else "confusion_matrix.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

# Naujos funkcijos vietos prognozėms:
def save_place_metrics(dates, maes, rmses, medaes, output_dir, prefix=""):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.plot(dates, maes, 'o-', label='MAE')
    plt.title("MAE pagal etapą")
    plt.xlabel("Data")
    plt.ylabel("MAE")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(dates, rmses, 'o-', label='RMSE', color='orange')
    plt.title("RMSE pagal etapą")
    plt.xlabel("Data")
    plt.ylabel("RMSE")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(dates, medaes, 'o-', label='MedAE', color='green')
    plt.title("MedAE pagal etapą")
    plt.xlabel("Data")
    plt.ylabel("MedAE")
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{prefix}_place_metrics.png" if prefix else "place_metrics.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def save_error_distribution(y_true, y_pred, output_dir, prefix=""):
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(8, 5))
    sns.histplot(errors, bins=30, kde=True, color="skyblue")
    plt.title("Prognozės klaidų pasiskirstymas")
    plt.xlabel("Absoliuti klaida (vietų skirtumas)")
    plt.ylabel("Sportininkių skaičius")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{prefix}_error_distribution.png" if prefix else "error_distribution.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()
