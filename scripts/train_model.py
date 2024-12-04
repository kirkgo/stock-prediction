#!/usr/bin/env python3

import argparse
import logging
import sys
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
import pandas as pd
from src.utils import evaluation
from src.utils.evaluation import (
    plot_training_metrics,
    plot_prediction_analysis,
    generate_metrics_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("Iniciando script de treinamento...")

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

logger.info(f"Project root configurado: {project_root}")

try:
    logger.info("Importando StockPredictor...")
    from src.models.lstm_model import StockPredictor
    logger.info("StockPredictor importado com sucesso")
except Exception as e:
    logger.error(f"Erro ao importar StockPredictor: {str(e)}")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    """Carrega configurações do arquivo YAML"""
    logger.info(f"Tentando carregar arquivo de configuração: {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info("Arquivo de configuração carregado com sucesso")
            return config
    except FileNotFoundError:
        logger.warning(
            f"Arquivo de configuração não encontrado: {config_path}")
        logger.info("Usando configurações padrão")
        return {
            "training": {
                "start_date": "2018-01-01",
                "end_date": "2024-07-20",
                "epochs": 100,
                "batch_size": 32,
                "validation_split": 0.2,
                "early_stopping_patience": 10
            },
            "model": {
                "sequence_length": 60,
                "lstm_units": [50, 50, 50],
                "dropout_rate": 0.2,
                "learning_rate": 0.001
            }
        }
    except Exception as e:
        logger.error(f"Erro ao carregar configurações: {str(e)}")
        sys.exit(1)


def plot_training_history(history, output_dir: str):
    """Plota e salva gráficos do histórico de treinamento"""
    logger.info("Gerando gráficos do histórico de treinamento...")

    try:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{output_dir}/training_history_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Gráficos salvos em: {plot_path}")
    except Exception as e:
        logger.error(f"Erro ao gerar gráficos: {str(e)}")


def main():
    logger.info("Iniciando função principal...")
    parser = argparse.ArgumentParser(
        description='Treina o modelo LSTM para previsão de ações')

    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Símbolo da ação (ex: AAPL)')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Caminho para o arquivo de configuração')
    parser.add_argument('--output-dir', type=str, default='models/saved_models',
                        help='Diretório para salvar o modelo treinado')

    args = parser.parse_args()
    logger.info(f"Argumentos recebidos: symbol={args.symbol}, config={
                args.config}, output_dir={args.output_dir}")

    try:
        # Carrega configurações
        config = load_config(args.config)

        logger.info(f"Iniciando treinamento para {args.symbol}")
        predictor = StockPredictor(
            symbol=args.symbol,
            start_date=config['training']['start_date'],
            end_date=config['training']['end_date']
        )

        # Treina o modelo
        logger.info(f"Iniciando treinamento com {
                    config['training']['epochs']} épocas...")
        history = predictor.train(
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            seq_length=config['model']['sequence_length']
        )
        logger.info("Treinamento concluído")

        # --- AQUI COMEÇAM AS CHAMADAS PARA GERAÇÃO DOS RELATÓRIOS ---

        # Gera visualizações do treinamento
        logger.info("Gerando visualizações do histórico de treinamento...")
        training_plots_path = plot_training_metrics(
            history.history, args.output_dir)
        logger.info(f"Visualizações do treinamento salvas em: {
                    training_plots_path}")

        # Avalia o modelo e gera predições
        logger.info("Avaliando o modelo...")
        data = predictor.fetch_data()
        _, X_test, _, y_test = predictor.prepare_sequences(data)
        metrics = predictor.evaluate(X_test, y_test)

        predictions = predictor.model.predict(X_test).flatten()
        y_test_inv = predictor.scaler.inverse_transform(
            np.concatenate(
                [y_test.reshape(-1, 1), np.zeros((len(y_test), 4))], axis=1)
        )[:, 0]

        # Gera datas para o período de teste
        dates = pd.date_range(
            start=config['training']['start_date'],
            periods=len(y_test),
            freq='D'
        )

        # Gera análise visual das predições
        logger.info("Gerando análise visual das predições...")
        prediction_plots_path = plot_prediction_analysis(
            predictions,
            y_test_inv,
            dates,
            args.output_dir
        )
        logger.info(f"Análise visual das predições salva em: {
                    prediction_plots_path}")

        # Gera relatório completo de métricas
        logger.info("Gerando relatório de métricas...")
        report_path = generate_metrics_report(
            metrics,
            predictions,
            y_test_inv,
            args.output_dir
        )
        logger.info(f"Relatório de métricas salvo em: {report_path}")

        # --- FIM DAS CHAMADAS DE RELATÓRIO ---

        # Salva o modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{args.output_dir}/{args.symbol}_model_{timestamp}.keras"
        logger.info(f"Salvando modelo em: {model_path}")
        predictor.save_model(model_path)

        logger.info("Script finalizado com sucesso!")

    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
