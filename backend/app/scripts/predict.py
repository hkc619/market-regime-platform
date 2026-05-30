import sys
sys.path.append('../../app')
from core.config import ModelConfig
from ml.inference import load_data, predict_proba

def predict():
    latest_60_feat, latest_regime = load_data(ModelConfig.data_path)
    predict_proba(latest_60_feat, latest_regime)

if __name__ == "__main__":
    predict()