import sys
import types
import os

os.environ.setdefault("STREAMLIT_SERVER_ENABLE_FILE_WATCHER", "false")

try:
    print("sitecustomize: loaded — installing torch.classes shim")
except Exception:
    pass

if "torch.classes" not in sys.modules:
    shim = types.ModuleType("torch.classes")
    shim.__path__ = []
    sys.modules["torch.classes"] = shim

try:
    if "torch" in sys.modules:
        _torch = sys.modules["torch"]
        if not hasattr(_torch, "classes") or _torch.classes is None:
            _tc = types.ModuleType("torch.classes")
            _tc.__path__ = []
            _torch.classes = _tc
except Exception:
    pass
