import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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