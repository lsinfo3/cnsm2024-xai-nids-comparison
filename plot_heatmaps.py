import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_annotated_heatmap(data, x_labels, y_labels, title, vmin, vmax, val1, val2, baseline_val, name, save_path=None):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(data, annot=False, cmap=sns.light_palette("teal", as_cmap=True), xticklabels=x_labels, yticklabels=y_labels, fmt=".2f", vmin=vmin, vmax=vmax, cbar=False, square=True)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    ax.axhline(y=0, color='k', linewidth=1)
    ax.axhline(y=data.shape[1], color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.axvline(x=data.shape[0], color='k', linewidth=1)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if val1 >= data[i, j] > baseline_val:
                ax.text(j + 0.5, i + 0.5, '•', color='darkblue', ha='center', va='center', fontsize=8)
            if val2 >= data[i, j] > val1:
                ax.text(j + 0.5, i + 0.5, '*', color='darkblue', ha='center', va='center', fontsize=14)
            if data[i, j] > val2:
                ax.text(j + 0.5, i + 0.5, '★', color='darkblue', ha='center', va='center', fontsize=14)

    plt.title(title, fontsize=18)
    
    
    if name == "EdgeIIoT": # rename for the plot; otherwise its too long
        ax.text(0.95,0.95, "IIOT", ha="right", va="top",size=16, color = "black",bbox=dict(facecolor='white', edgecolor='black',boxstyle='round',alpha=.75),transform = ax.transAxes)
    else:
        ax.text(0.95,0.95, name, ha="right", va="top",size=16, color = "black",bbox=dict(facecolor='white', edgecolor='black',boxstyle='round',alpha=.75),transform = ax.transAxes)

    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_all_matrices(file_path, dataset_name, selection_method, complexity):
    all_matrices = joblib.load(file_path) # contains the 3 consensus matrices

    unordered_matrix = all_matrices[(selection_method, "unordered", False)]
    ordered_matrix = all_matrices[(selection_method, "ordered", False)]
    agreement_matrix = all_matrices[(selection_method, "sign", False)]

    # calculate the avg. matrices over all samples (axis=2)
    average_unordered_comparison = np.mean(unordered_matrix, axis=2)
    average_ordered_comparison = np.mean(ordered_matrix, axis=2)
    average_agreement_matrix = np.mean(agreement_matrix, axis=2)

    # labels for the heatmaps
    # IMPORTANT: same order as in the explainer script!!
    valid_labels = [
        'SHAP DT', 'LIME DT', 'TI DT',
        'SHAP RF', 'LIME RF', 'TI RF',
        'SHAP LGBM', 'LIME LGBM',
        'SHAP SLP', 'LIME SLP',
        'SHAP MLP', 'LIME MLP',
        'Sal. SLP', 'DLIFT SLP', 'IG SLP',
        'Sal. MLP', 'DLIFT MLP', 'IG MLP'
    ]

    # plottttt actual heatmaps
    plot_annotated_heatmap(
        average_agreement_matrix, valid_labels, valid_labels, 
        "", vmin=0, vmax=100, val2=75, val1=50, baseline_val=33, name=dataset_name,
        save_path=f"sign_comparison_heatmap_{dataset_name}_{selection_method}_binary.pdf"
    )

    plot_annotated_heatmap(
        average_unordered_comparison, valid_labels, valid_labels, 
        "", vmin=0, vmax=5, val2=4, val1=3, baseline_val=2.5, name=dataset_name,
        save_path=f"unordered_comparison_heatmap_{dataset_name}_{selection_method}_binary.pdf"
    )

    plot_annotated_heatmap(
        average_ordered_comparison, valid_labels, valid_labels, 
        "", vmin=0, vmax=5, val2=1.0, val1=0.5, baseline_val=0.113, name=dataset_name,
        save_path=f"ordered_comparison_heatmap_{dataset_name}_{selection_method}_binary.pdf"
    )


if __name__ == "__main__":
    dataset_name = "EdgeIIoT"
    file_path = os.path.join("data",'all_matrices_'+dataset_name+'.pkl')
    for selection_method in ["impurity"]:
        for complexity in [False]:
            plot_all_matrices(file_path, dataset_name, selection_method, complexity)
