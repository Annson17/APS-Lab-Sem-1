from data_loader import load_dataset
from modeling import compare_thresholds, evaluate_model, generate_report_table, print_metrics_for_thresholds, train_model
from visualization import plot_class_distribution


def main():
    X, y = load_dataset()
    plot_class_distribution(y)

    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    compare_thresholds(model, X_test, y_test)
    print_metrics_for_thresholds(model, X_test, y_test)
    generate_report_table(model, X_test, y_test)


if __name__ == "__main__":
    main()
