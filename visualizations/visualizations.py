import os
import matplotlib
matplotlib.use("Agg")  # Užtikrina, kad grafikai nebūtų rodomi GUI režimu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score


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

def plot_feature_importance(model, feature_names, output_dir, prefix):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(10, len(feature_names))
    plt.figure(figsize=(12, 6))
    plt.title(f'Top {top_n} svarbiausių požymių')
    plt.bar(range(top_n), importances[indices][:top_n], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_feature_importance.png"))
    plt.close()
    print("\n🔍 Top 10 svarbiausių požymių:")
    for i in range(top_n):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


def plot_permutation_importance(model, X, y, feature_names, output_dir, prefix):

    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    top_n = min(10, len(feature_names))
    plt.figure(figsize=(12, 6))
    plt.title(f'Top {top_n} požymių pagal permutacijos svarbą')
    plt.boxplot(perm_importance.importances[sorted_idx][:top_n].T, 
                vert=False, labels=[feature_names[i] for i in sorted_idx[:top_n]])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_permutation_importance.png"))
    plt.close()
    print("\n🔄 Top 10 požymių pagal permutacijos svarbą:")
    for i in range(top_n):
        idx = sorted_idx[i]
        print(f"{i+1}. {feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f} ± {perm_importance.importances_std[idx]:.4f}")

def plot_n_estimators_performance(X, y, output_dir, prefix):

    n_estimators_range = range(10, 301, 20)
    mse_scores = []
    for n_est in n_estimators_range:
        model = RandomForestRegressor(n_estimators=n_est, random_state=42)
        mse = -np.mean(cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error'))
        mse_scores.append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, mse_scores, marker='o')
    plt.title('MSE priklausomybė nuo n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    
    best_idx = np.argmin(mse_scores)
    best_n_estimators = n_estimators_range[best_idx]
    best_mse = mse_scores[best_idx]
    
    plt.scatter(best_n_estimators, best_mse, c='red', s=100, zorder=10)
    plt.annotate(f'Optimalu: {best_n_estimators}',
                 (best_n_estimators, best_mse),
                 xytext=(5, -15),
                 textcoords='offset points')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_n_estimators.png"))
    plt.close()
    
    print(f"\n Optimalus n_estimators: {best_n_estimators} (MSE: {best_mse:.4f})")
    return best_n_estimators

def plot_prediction_distribution(raw_preds, y_true, output_dir, prefix, competition_name):

    plt.figure(figsize=(10, 6))
    positives = raw_preds[y_true == 1]
    negatives = raw_preds[y_true == 0]
    
    plt.hist(positives, bins=20, alpha=0.7, label='Dalyvavo', color='green')
    plt.hist(negatives, bins=20, alpha=0.7, label='Nedalyvavo', color='red')
    
    plt.title(f'Prognozių pasiskirstymas - {competition_name}')
    plt.xlabel('Prognozuota tikimybė dalyvauti')
    plt.ylabel('Kiekis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_prediction_dist_{competition_name.replace(' ', '_')}.png"))
    plt.close()