import os
import torch
import asyncio
import logging
from pathlib import Path
from loguru import logger
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

_orig_read_text = Path.read_text
def _read_text_utf8(self, encoding=None, errors=None):
    if encoding is None:
        encoding = 'utf-8'
    return _orig_read_text(self, encoding=encoding, errors=errors)
Path.read_text = _read_text_utf8

_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs or kwargs['weights_only'] != True:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# 在 Windows 下使用 SelectorEventLoopPolicy，避免关闭时的 WinError 10054 噪声
try:
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        logging.getLogger("asyncio").setLevel(logging.ERROR)
except Exception as e:
    logger.warning(f"设置 Windows 事件循环策略失败：{e}")

# 抑制第三方库 fastrtc 的周期性 WARNING 提示（如 60s 帧处理超时）
try:
    logging.getLogger("fastrtc").setLevel(logging.ERROR)
    logging.getLogger("fastrtc.utils").setLevel(logging.ERROR)
except Exception:
    pass
