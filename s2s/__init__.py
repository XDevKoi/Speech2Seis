"""Speech2Seis (S2S): cross-modal transfer of a pretrained speech model to seismic monitoring."""

from .backbone import Speech2Seis
from .common import (
    WAV2VEC2_DOWNLOAD_URL,
    ScaledActivation,
    lora_setting,
    resolve_wav2vec2_path,
)
from .heads import (
    HeadBAZDisLocal,
    HeadBAZLocal,
    HeadClassificationLocal,
    HeadDetectionPicking,
    HeadRegressionLocal,
)
from .losses import BAZDisLoss, BCELoss, CELoss
from .models import (
    MODELS,
    TASK_SPECS,
    S2S_baz,
    S2S_bazdis,
    S2S_dis,
    S2S_dpk,
    S2S_pmp,
    build_model,
    load_checkpoint,
)
from .preprocess import cut_window_p_centered, fit_length, normalize, prepare_p_centered

__version__ = "0.1.0"

__all__ = [
    "Speech2Seis",
    "S2S_dpk",
    "S2S_pmp",
    "S2S_baz",
    "S2S_dis",
    "S2S_bazdis",
    "MODELS",
    "TASK_SPECS",
    "build_model",
    "load_checkpoint",
    "HeadDetectionPicking",
    "HeadClassificationLocal",
    "HeadRegressionLocal",
    "HeadBAZLocal",
    "HeadBAZDisLocal",
    "ScaledActivation",
    "BCELoss",
    "CELoss",
    "BAZDisLoss",
    "normalize",
    "cut_window_p_centered",
    "prepare_p_centered",
    "fit_length",
    "lora_setting",
    "resolve_wav2vec2_path",
    "WAV2VEC2_DOWNLOAD_URL",
    "__version__",
]
