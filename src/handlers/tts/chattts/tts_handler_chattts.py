import sys
import io
import os
import re
import time
from typing import Dict, Optional, cast
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
import torch
import yaml
import ChatTTS
# Patch ChatTTS to support .pt files
import ChatTTS.utils.io
import ChatTTS.core
import ChatTTS.model.dvae
# Try to import other modules that might use load_safetensors
try:
    import ChatTTS.model.gpt
    import ChatTTS.model.embed
except ImportError:
    pass

original_load_safetensors = ChatTTS.utils.io.load_safetensors

def patched_load_safetensors(filename: str):
    if filename.endswith('.pt'):
        logger.info(f"Loading .pt file: {filename}")
        return torch.load(filename, map_location='cuda:0')
    return original_load_safetensors(filename)

# Apply patch to all places
ChatTTS.utils.io.load_safetensors = patched_load_safetensors
if hasattr(ChatTTS.core, 'load_safetensors'):
    ChatTTS.core.load_safetensors = patched_load_safetensors
if hasattr(ChatTTS.model.dvae, 'load_safetensors'):
    ChatTTS.model.dvae.load_safetensors = patched_load_safetensors
if 'ChatTTS.model.gpt' in sys.modules and hasattr(sys.modules['ChatTTS.model.gpt'], 'load_safetensors'):
    sys.modules['ChatTTS.model.gpt'].load_safetensors = patched_load_safetensors
if 'ChatTTS.model.embed' in sys.modules and hasattr(sys.modules['ChatTTS.model.embed'], 'load_safetensors'):
    sys.modules['ChatTTS.model.embed'].load_safetensors = patched_load_safetensors

from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from engine_utils.directory_info import DirectoryInfo

class ChatTTSConfig(HandlerBaseConfigModel, BaseModel):
    temperature: float = Field(default=0.3)
    top_P: float = Field(default=0.7)
    top_K: int = Field(default=20)
    seed: int = Field(default=42)
    audio_speed: int = Field(default=5)

class ChatTTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None
        self.spk_emb = None
        self.config: Optional[ChatTTSConfig] = None

class HandlerChatTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.chat = None
        self.sample_rate = 24000
        self.default_spk_emb = None
        self.config: Optional[ChatTTSConfig] = None
        
    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=ChatTTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        config = cast(ChatTTSConfig, handler_config)
        self.config = config
        
        logger.info("Loading ChatTTS model...")
        self.chat = ChatTTS.Chat()
        
        # Load local models from 'models/AI-ModelScope/ChatTTS'
        model_root = os.path.join(DirectoryInfo.get_project_dir(), 'models/AI-ModelScope/ChatTTS')
        
        # Manually construct load_args to use .safetensors and ensure compatibility
        load_args = {
            'vocos_ckpt_path': os.path.join(model_root, 'asset', 'Vocos.safetensors'),
            'dvae_ckpt_path': os.path.join(model_root, 'asset', 'DVAE.safetensors'),
             'gpt_ckpt_path': os.path.join(model_root, 'asset', 'gpt'),
             'decoder_ckpt_path': os.path.join(model_root, 'asset', 'Decoder.safetensors'),
             'tokenizer_path': os.path.join(model_root, 'asset', 'tokenizer'),
             'embed_path': os.path.join(model_root, 'asset', 'Embed.safetensors'),
        }
        
        # Call _load directly to bypass check_all_assets
        logger.info(f"Calling ChatTTS._load with args: {load_args}")
        self.chat._load(**load_args, compile=False) 
        
        logger.info("ChatTTS model loaded.")
        
        # Sample a fixed speaker embedding
        torch.manual_seed(config.seed)
        self.default_spk_emb = self.chat.sample_random_speaker()

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, ChatTTSConfig):
            handler_config = self.config if self.config else ChatTTSConfig()
            
        context = ChatTTSContext(session_context.session_info.session_id)
        context.config = handler_config
        context.input_text = ''
        
        if self.config and handler_config.seed == self.config.seed and self.default_spk_emb is not None:
            context.spk_emb = self.default_spk_emb
        else:
            torch.manual_seed(handler_config.seed)
            context.spk_emb = self.chat.sample_random_speaker()
        
        if context.dump_audio:
             dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp',
                                            f"dump_avatar_audio_{context.session_id}_{time.localtime().tm_hour}_{time.localtime().tm_min}.pcm")
             context.audio_dump_file = open(dump_file_path, "wb")
        return context
    
    def start_context(self, session_context, context: HandlerContext):
        pass

    def filter_text(self, text):
        # Normalize Chinese punctuation to English
        text = text.replace("，", ",").replace("。", ".").replace("？", "?").replace("！", "!").replace("：", ":").replace("；", ";")
        pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!? ]"
        filtered_text = re.sub(pattern, "", text)
        return filtered_text

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(ChatTTSContext, context)
        
        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
            
        speech_id = inputs.data.get_meta("speech_id")
        if (speech_id is None):
            speech_id = context.session_id

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            context.input_text += self.filter_text(text)

        text_end = inputs.data.get_meta("avatar_text_end", False)
        
        sentences_to_process = []
        
        if not text_end:
            sentences = re.split(r'(?<=[,.~!?，。！？])', context.input_text)
            if len(sentences) > 1:
                complete_sentences = sentences[:-1]
                context.input_text = sentences[-1]
                sentences_to_process = complete_sentences
        else:
             if context.input_text is not None and len(context.input_text.strip()) > 0:
                 sentences_to_process = [context.input_text]
             context.input_text = ''

        for sentence in sentences_to_process:
            if len(sentence.strip()) < 1:
                continue
            logger.info('ChatTTS Generating: ' + sentence)
            
            # Infer
            prompt = f"[speed_{context.config.audio_speed}]"
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb = context.spk_emb,
                temperature = context.config.temperature,
                top_P = context.config.top_P,
                top_K = context.config.top_K,
                prompt = prompt,
            )
            
            wavs = self.chat.infer([sentence], params_infer_code=params_infer_code, use_decoder=True)
            
            if wavs and len(wavs) > 0:
                audio_data = wavs[0]
                # Check shape, expecting (channels, samples) or (samples,)
                # ChatTTS usually returns (samples,) or (1, samples)
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()
                
                if len(audio_data.shape) == 1:
                    audio_data = audio_data[np.newaxis, ...]
                
                output = DataBundle(output_definition)
                output.set_main_data(audio_data)
                output.add_meta("avatar_speech_end", False)
                output.add_meta("speech_id", speech_id)
                context.submit_data(output)
        
        if text_end:
            output = DataBundle(output_definition)
            output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
            output.add_meta("avatar_speech_end", True)
            output.add_meta("speech_id", speech_id)
            context.submit_data(output)
            logger.info(f"语音结束")

    def destroy_context(self, context: HandlerContext):
        context = cast(ChatTTSContext, context)
        logger.info('销毁上下文')
