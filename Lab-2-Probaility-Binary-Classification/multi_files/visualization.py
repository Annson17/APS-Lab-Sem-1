import matplotlib.pyplot as plt
import pandas as pd


def plot_class_distribution(y):
    class_counts = y.value_counts().sort_index()

    class_distribution = pd.DataFrame({
        "Class": ["malignant", "benign"],
        "Count": class_counts.values,
        "Probability": class_counts.values / len(y),
    })

    print("\nClass Distribution:\n")
    print(class_distribution)

    class_distribution.plot(
        x="Class",
        y="Count",
        kind="bar",
        color=["red", "green"],
        legend=False,
    )
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
