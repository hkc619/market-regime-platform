FEATURE_WARMUP_DAYS = 252
MODEL_SEQUENCE_LENGTH = 60
RAW_LOOKBACK_DAYS = FEATURE_WARMUP_DAYS + MODEL_SEQUENCE_LENGTH

MODEL_VERSION = "cnn_gru_v1"

REGIME_LABELS = {
    0: "Trending-Down",
    1: "Transition-Down",
    2: "Transition-Up",
    3: "Trending-Up",
}