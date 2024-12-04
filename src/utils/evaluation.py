import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from typing import Dict, List
import pandas as pd


def plot_training_metrics(history: Dict, output_dir: str) -> str:
    """
    Plota métricas do treinamento (loss e métricas de validação)
    """
    plt.style.use('default')
    plt.figure(figsize=(15, 10))

    # Plot de Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot de MAE
    plt.subplot(2, 2, 2)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    # Plot de MAPE
    if 'mape' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['mape'], label='Training MAPE')
        plt.plot(history['val_mape'], label='Validation MAPE')
        plt.title('Mean Absolute Percentage Error Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('MAPE (%)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    # Salva o plot
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'training_metrics_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath


def plot_prediction_analysis(predictions: np.ndarray,
                             actual_values: np.ndarray,
                             dates: List,
                             output_dir: str) -> str:
    """
    Gera análise visual detalhada das predições vs valores reais
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))

    # 1. Série temporal de previsões vs valores reais
    plt.subplot(3, 2, 1)
    plt.plot(dates, actual_values, label='Actual Values', color='blue')
    plt.plot(dates, predictions, label='Predictions',
             color='red', linestyle='--')
    plt.title('Stock Price: Predicted vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    # 2. Scatter plot de correlação
    plt.subplot(3, 2, 2)
    plt.scatter(actual_values, predictions, alpha=0.5)
    plt.plot([actual_values.min(), actual_values.max()],
             [actual_values.min(), actual_values.max()],
             'r--', lw=2)
    plt.title('Correlation Plot: Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)

    # 3. Distribuição dos erros
    errors = predictions - actual_values
    plt.subplot(3, 2, 3)
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # 4. Error vs Actual Value
    plt.subplot(3, 2, 4)
    plt.scatter(actual_values, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Error vs Actual Value')
    plt.xlabel('Actual Value')
    plt.ylabel('Error')
    plt.grid(True)

    # 5. Percentage Error Over Time
    percent_errors = (errors / actual_values) * 100
    plt.subplot(3, 2, 5)
    plt.plot(dates, percent_errors)
    plt.title('Percentage Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Percentage Error (%)')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 6. Box plot dos erros
    plt.subplot(3, 2, 6)
    plt.boxplot(percent_errors)
    plt.title('Box Plot of Percentage Errors')
    plt.ylabel('Percentage Error (%)')
    plt.grid(True)

    plt.tight_layout()

    # Salva o plot
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'prediction_analysis_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath


def generate_metrics_report(metrics: Dict, predictions: np.ndarray,
                            actual_values: np.ndarray, output_dir: str) -> str:
    """
    Gera um relatório detalhado de métricas em formato JSON
    """
    # Calcula métricas adicionais
    error = predictions - actual_values
    percentage_error = (error / actual_values) * 100

    extended_metrics = {
        "basic_metrics": {
            "mae": metrics['MAE'],
            "rmse": metrics['RMSE'],
            "mape": metrics['MAPE']
        },
        "error_statistics": {
            "mean_error": float(np.mean(error)),
            "std_error": float(np.std(error)),
            "median_error": float(np.median(error)),
            "min_error": float(np.min(error)),
            "max_error": float(np.max(error))
        },
        "percentage_error_statistics": {
            "mean_percentage_error": float(np.mean(percentage_error)),
            "std_percentage_error": float(np.std(percentage_error)),
            "median_percentage_error": float(np.median(percentage_error)),
            "min_percentage_error": float(np.min(percentage_error)),
            "max_percentage_error": float(np.max(percentage_error))
        },
        "prediction_statistics": {
            "actual_mean": float(np.mean(actual_values)),
            "predicted_mean": float(np.mean(predictions)),
            "actual_std": float(np.std(actual_values)),
            "predicted_std": float(np.std(predictions)),
            "correlation": float(np.corrcoef(actual_values, predictions)[0, 1])
        },
        "timestamp": datetime.now().isoformat(),
        "interpretation": {
            "model_performance": _interpret_performance(metrics),
            "recommendations": _generate_recommendations(metrics)
        }
    }

    # Salva o relatório
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'metrics_report_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(extended_metrics, f, indent=4)

    return filepath


def _interpret_performance(metrics: Dict) -> str:
    """
    Interpreta o desempenho do modelo baseado nas métricas
    """
    mape = metrics['MAPE']

    if mape < 5:
        return "Excelente: O modelo apresenta alta precisão com erro percentual médio menor que 5%"
    elif mape < 10:
        return "Bom: O modelo apresenta boa precisão com erro percentual médio menor que 10%"
    elif mape < 15:
        return "Regular: O modelo apresenta precisão moderada com erro percentual médio menor que 15%"
    else:
        return "Necessita Melhorias: O modelo apresenta erro percentual médio superior a 15%"


def _generate_recommendations(metrics: Dict) -> List[str]:
    """
    Gera recomendações baseadas nas métricas do modelo
    """
    recommendations = []
    mape = metrics['MAPE']

    if mape > 15:
        recommendations.extend([
            "Considerar aumentar o tamanho do conjunto de treinamento",
            "Avaliar a inclusão de features adicionais",
            "Experimentar diferentes arquiteturas de rede neural"
        ])

    if mape > 10:
        recommendations.extend([
            "Avaliar a necessidade de mais épocas de treinamento",
            "Considerar ajustes nos hiperparâmetros do modelo"
        ])

    if mape > 5:
        recommendations.append(
            "Verificar se há sazonalidade não capturada pelo modelo"
        )

    if not recommendations:
        recommendations.append(
            "Modelo apresenta boa performance. Manter monitoramento contínuo."
        )

    return recommendations
