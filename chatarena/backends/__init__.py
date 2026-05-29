from ..config import BackendConfig

from .base import IntelligenceBackend
from .openai import OpenAIChat
from .qwen import QwenChat
from .human import Human
from .gemini import GeminiChat

# 尝试导入 transformers backend，如果失败则跳过
try:
    from .hf_transformers import TransformersConversational
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    TransformersConversational = None

ALL_BACKENDS = [
    Human,
    OpenAIChat,
    QwenChat,
    GeminiChat,
]

if HAS_TRANSFORMERS:
    ALL_BACKENDS.append(TransformersConversational)

BACKEND_REGISTRY = {backend.type_name: backend for backend in ALL_BACKENDS}


# Load a backend from a config dictionary
def load_backend(config: BackendConfig, args=None):
    try:
        backend_cls = BACKEND_REGISTRY[config.backend_type]
    except KeyError:
        raise ValueError(f"Unknown backend type: {config.backend_type}")

    backend = backend_cls.from_config(config, args)
    return backend
