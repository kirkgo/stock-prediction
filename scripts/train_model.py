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

    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Símbolo da ação (ex: AAPL)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Caminho para o arquivo de configuração'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/saved_models',
        help='Diretório para salvar o modelo treinado'
    )

    args = parser.parse_args()
    logger.info(f"Argumentos recebidos: symbol={args.symbol}, config={
                args.config}, output_dir={args.output_dir}")

    try:
        # Carrega configurações
        logger.info("Carregando configurações...")
        config = load_config(args.config)

        logger.info(f"Iniciando treinamento para {args.symbol}")
        logger.info(f"Período de treinamento: {config['training']['start_date']} até {
                    config['training']['end_date']}")

        # Cria diretório de saída
        logger.info(f"Criando diretório de saída: {args.output_dir}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Inicializa o modelo
        logger.info("Inicializando StockPredictor...")
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

        # Plota histórico
        logger.info("Gerando visualizações do treinamento...")
        plot_training_history(history, args.output_dir)

        # Avalia o modelo
        logger.info("Avaliando o modelo...")
        data = predictor.fetch_data()
        _, X_test, _, y_test = predictor.prepare_sequences(data)
        metrics = predictor.evaluate(X_test, y_test)

        logger.info("Métricas de avaliação:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Salvar relatório de métricas
        logger.info("Salvando relatório de métricas...")
        report_path = evaluation.save_metrics_report(
            metrics, args.output_dir)
        logger.info(f"Relatório de métricas salvo em: {report_path}")

        # Plotar gráficos de avaliação
        logger.info("Plotando gráficos de avaliação...")
        predictions = predictor.model.predict(X_test).flatten()
        y_test_inv = predictor.scaler.inverse_transform(
            np.concatenate(
                [y_test.reshape(-1, 1), np.zeros((len(y_test), 4))], axis=1)
        )[:, 0]
        dates = pd.date_range(
            start=config['training']['start_date'], periods=len(y_test), freq='D')
        plot_path = evaluation.plot_metrics(
            predictions, y_test_inv, dates, args.output_dir)
        logger.info(f"Gráficos de avaliação salvos em: {plot_path}")

        # Salva o modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{
            args.output_dir}/{args.symbol}_model_{timestamp}.keras"
        logger.info(f"Salvando modelo em: {model_path}")
        predictor.save_model(model_path)

        logger.info("Script finalizado com sucesso!")

    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# def main():
#     logger.info("Iniciando função principal...")
#     parser = argparse.ArgumentParser(
#         description='Treina o modelo LSTM para previsão de ações')

#     parser.add_argument(
#         '--symbol',
#         type=str,
#         default='AAPL',
#         help='Símbolo da ação (ex: AAPL)'
#     )

#     parser.add_argument(
#         '--config',
#         type=str,
#         default='configs/training_config.yaml',
#         help='Caminho para o arquivo de configuração'
#     )

#     parser.add_argument(
#         '--output-dir',
#         type=str,
#         default='models/saved_models',
#         help='Diretório para salvar o modelo treinado'
#     )

#     args = parser.parse_args()
#     logger.info(f"Argumentos recebidos: symbol={args.symbol}, config={
#                 args.config}, output_dir={args.output_dir}")

#     try:
#         logger.info("Carregando configurações...")
#         config = load_config(args.config)

#         logger.info(f"Iniciando treinamento para {args.symbol}")
#         logger.info(f"Período de treinamento: {config['training']['start_date']} até {
#                     config['training']['end_date']}")

#         logger.info(f"Criando diretório de saída: {args.output_dir}")
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)

#         logger.info("Inicializando StockPredictor...")
#         predictor = StockPredictor(
#             symbol=args.symbol,
#             start_date=config['training']['start_date'],
#             end_date=config['training']['end_date']
#         )

#         logger.info(f"Iniciando treinamento com {
#                     config['training']['epochs']} épocas...")
#         history = predictor.train(
#             epochs=config['training']['epochs'],
#             batch_size=config['training']['batch_size'],
#             seq_length=config['model']['sequence_length']
#         )
#         logger.info("Treinamento concluído")

#         logger.info("Gerando visualizações do treinamento...")
#         plot_training_history(history, args.output_dir)

#         logger.info("Avaliando o modelo...")
#         data = predictor.fetch_data()
#         _, X_test, _, y_test = predictor.prepare_sequences(data)
#         metrics = predictor.evaluate(X_test, y_test)

#         logger.info("Métricas de avaliação:")
#         for metric, value in metrics.items():
#             logger.info(f"{metric}: {value:.4f}")

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         model_path = f"{args.output_dir}/{args.symbol}_model_{timestamp}.keras"
#         logger.info(f"Salvando modelo em: {model_path}")
#         predictor.save_model(model_path)

#         logger.info("Fazendo predições de teste...")
#         predictions = predictor.predict(days=30)
#         logger.info(f"Primeiras 5 predições: {predictions[:5]}")

#         logger.info("Script finalizado com sucesso!")

#     except Exception as e:
#         logger.error(f"Erro durante a execução: {str(e)}", exc_info=True)
#         sys.exit(1)


# if __name__ == "__main__":
#     main()
