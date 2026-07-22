import numpy as np

import pickle

from app.core.logging import get_logger
logger = get_logger("inference_service")

def predict_proba(bundle, device, model, scaler):
    import torch
    
    logger.info("Building inference sequence | ticker=%s", bundle.ticker)

    with open(f"{scaler}", "rb") as f:
        scaler = pickle.load(f)
    
    latest_60_scaled = scaler.transform(bundle.latest_60_feat)
    latest_20_scaled = latest_60_scaled[-20:]

    Xm_tensor = torch.tensor(latest_60_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    Xs_tensor = torch.tensor(latest_20_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    r_tensor = torch.tensor([bundle.latest_regime], dtype=torch.long).to(device)

    logger.info("Running model inference | ticker=%s", bundle.ticker)
    with torch.no_grad():
        logits = model(Xs_tensor, Xm_tensor, r_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        predicted_class = np.argmax(probabilities)
        confidence_value = float(probabilities[0][predicted_class])


    STATE_NAMES = {0: "Trending-Down", 1: "Transition-Down", 2: "Transition-Up", 3: "Trending-Up"}
    
    result = {
        "ticker": bundle.ticker,
        "last day: ": bundle.end_date.strftime("%Y-%m-%d"),
        "confidence": confidence_value,
        "predicted_class": predicted_class,
        "predicted_regime": STATE_NAMES[predicted_class],
        "probabilities": {
            STATE_NAMES[0]: float(probabilities[0][0]),
            STATE_NAMES[1]: float(probabilities[0][1]),
            STATE_NAMES[2]: float(probabilities[0][2]),
            STATE_NAMES[3]: float(probabilities[0][3]),
        }
    }

    logger.info(
        "Inference pipeline completed | ticker=%s | predicted_state=%s | confidence=%d ",
        bundle.ticker, STATE_NAMES[predicted_class], confidence_value
    )
    
    return result
