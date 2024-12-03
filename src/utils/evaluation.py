import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os


def plot_metrics(predictions, actual_values, dates, output_dir):
    """
    Plot evaluation metrics and predictions vs actual values
    """
    plt.figure(figsize=(15, 10))

    # Subplot para preços
    plt.subplot(2, 1, 1)
    plt.plot(dates, actual_values, label='Actual Values', color='blue')
    plt.plot(dates, predictions, label='Predictions',
             color='red', linestyle='--')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)

    # Subplot para erro
    plt.subplot(2, 1, 2)
    error = np.abs(predictions - actual_values)
    plt.bar(dates, error, color='orange', alpha=0.6)
    plt.title('Absolute Error')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Salvar o gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{output_dir}/evaluation_metrics_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def save_metrics_report(metrics, output_dir):
    """
    Save metrics report to a JSON file
    """
    report = {
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "interpretation": {
            "MAE": "Average absolute difference between predicted and actual values",
            "RMSE": "Root mean square error (penalizes larger errors more heavily)",
            "MAPE": "Average percentage difference between predicted and actual values"
        },
        "summary": {
            "MAE": f"{metrics['MAE']:.2f}",
            "RMSE": f"{metrics['RMSE']:.2f}",
            "MAPE": f"{metrics['MAPE']:.2f}%"
        }
    }

    # Adicionar análise qualitativa
    mae_threshold = 10  # Ajuste o threshold conforme necessário
    mape_threshold = 15  # Ajuste o threshold conforme necessário

    if metrics['MAE'] < mae_threshold and metrics['MAPE'] < mape_threshold:
        performance = "Good"
    elif metrics['MAE'] < mae_threshold * 2 and metrics['MAPE'] < mape_threshold * 2:
        performance = "Moderate"
    else:
        performance = "Needs Improvement"

    report["performance_assessment"] = {
        "overall_performance": performance,
        "recommendations": [
            "Consider increasing training data if performance is poor",
            "Adjust model hyperparameters if error is high",
            "Review feature selection if predictions are off"
        ]
    }

    # Salvar relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_dir}/metrics_report_{timestamp}.json"

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    return report_path
