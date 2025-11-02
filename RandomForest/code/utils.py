from sklearn.tree import plot_tree
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def plot_feature_importances(model, feature_names, top_n=20, fname=None):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    imp = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(6, max(3, top_n*0.3)))
    sns.barplot(x=imp.values, y=imp.index, orient='h')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
        plt.close()
    else:
        plt.show()


def view_tree_from_rf(rf_classifier, features, class_names):
    tree = rf_classifier.estimators_[0]
    plt.figure(figsize=(20,10))  # Set figure size to make the tree more readable
    plot_tree(tree, 
            feature_names=features,  # Use the feature names from the dataset
            class_names=class_names,  # Use class names (species names)
            filled=True,              # Fill nodes with colors for better visualization
            rounded=True)             # Rounded edges for nodes
    plt.title("Decision Tree from the Random Forest")
    plt.show()
    plt.savefig(f"{PIC_PATH}/{rf_classifier.name}{timestamp}.png")