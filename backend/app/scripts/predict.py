import sys
sys.path.append('../../app')

import json

from core.config import ModelConfig
from ml.inference import load_data, predict_proba

def predict():
    latest_60_feat, latest_regime = load_data(ModelConfig.data_path)
    data = predict_proba(latest_60_feat, latest_regime)
    with open('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/output/output.json',
               'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    predict()