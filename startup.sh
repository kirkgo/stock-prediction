#!/bin/bash

# Verifica se existem modelos treinados
if [ ! "$(ls -A /app/models/saved_models)" ]; then
    echo "No trained models found. Training models..."
    python scripts/train_model.py --symbol AAPL --output-dir models/saved_models
fi

# Inicia a API usando o caminho completo do uvicorn
exec ~/.local/bin/uvicorn src.api.main:app --host 0.0.0.0 --port ${API_PORT:-8001}
