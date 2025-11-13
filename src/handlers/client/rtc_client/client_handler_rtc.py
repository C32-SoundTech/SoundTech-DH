import asyncio
import time
from pathlib import Path
from typing import Dict, Optional, cast, Union, Tuple
from uuid import uuid4

from loguru import logger

from engine_utils.directory_info import DirectoryInfo
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import gradio
import numpy as np
from fastapi import FastAPI
# noinspection PyPackageRequirements
from fastrtc import Stream

from pydantic import BaseModel, Field

from chat_engine.common.client_handler_base import ClientHandlerBase, ClientSessionDelegate
from chat_engine.common.engine_channel_type import EngineChannelType
from chat_engine.common.handler_base import HandlerDataInfo, HandlerDetail, HandlerBaseInfo
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, VariableSize, \
    DataBundle
from service.rtc_service.rtc_provider import RTCProvider
from service.rtc_service.rtc_stream import RtcStream


class RtcClientSessionDelegate(ClientSessionDelegate):
    def __init__(self):
        self.timestamp_generator = None
        self.data_submitter = None
        self.shared_states = None
        self.output_queues = {
            EngineChannelType.AUDIO: asyncio.Queue(),
            EngineChannelType.VIDEO: asyncio.Queue(),
            EngineChannelType.TEXT: asyncio.Queue(),
        }
        self.input_data_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self.modality_mapping = {
            EngineChannelType.AUDIO: ChatDataType.MIC_AUDIO,
            EngineChannelType.VIDEO: ChatDataType.CAMERA_VIDEO,
            EngineChannelType.TEXT: ChatDataType.HUMAN_TEXT,
        }

    async def get_data(self, modality: EngineChannelType, timeout: Optional[float] = 0.1) -> Optional[ChatData]:
        data_queue = self.output_queues.get(modality)
        if data_queue is None:
            return None
        if timeout is not None and timeout > 0:
            try:
                data = await asyncio.wait_for(data_queue.get(), timeout)
            except asyncio.TimeoutError:
                return None
        else:
            data = await data_queue.get()
        return data

    def put_data(self, modality: EngineChannelType, data: Union[np.ndarray, str],
                 timestamp: Optional[Tuple[int, int]] = None, samplerate: Optional[int] = None, loopback: bool = False):
        if timestamp is None:
            timestamp = self.get_timestamp()
        if self.data_submitter is None:
            return
        definition = self.input_data_definitions.get(modality)
        chat_data_type = self.modality_mapping.get(modality)
        if chat_data_type is None or definition is None:
            return
        data_bundle = DataBundle(definition)
        if modality == EngineChannelType.AUDIO:
            data_bundle.set_main_data(data.squeeze()[np.newaxis, ...])
        elif modality == EngineChannelType.VIDEO:
            data_bundle.set_main_data(data[np.newaxis, ...])
        elif modality == EngineChannelType.TEXT:
            data_bundle.add_meta('human_text_end', True)
            data_bundle.add_meta('speech_id', str(uuid4()))
            data_bundle.set_main_data(data)
        else:
            return
        chat_data = ChatData(
            source="client",
            type=chat_data_type,
            data=data_bundle,
            timestamp=timestamp,
        )
        self.data_submitter.submit(chat_data)
        if loopback:
            self.output_queues[modality].put_nowait(chat_data)

    def get_timestamp(self):
        return self.timestamp_generator()

    def emit_signal(self, signal: ChatSignal):
        pass

    def clear_data(self):
        for data_queue in self.output_queues.values():
            while not data_queue.empty():
                data_queue.get_nowait()


class ClientRtcConfigModel(HandlerBaseConfigModel, BaseModel):
    connection_ttl: int = Field(default=900)
    turn_config: Optional[Dict] = Field(default=None)


class TextPayload(BaseModel):
    text: str


class ClientRtcContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[ClientRtcConfigModel] = None
        self.client_session_delegate: Optional[RtcClientSessionDelegate] = None


class ClientHandlerRtc(ClientHandlerBase):
    def __init__(self):
        super().__init__()
        self.engine_config = None
        self.handler_config = None
        self.rtc_streamer_factory: Optional[RtcStream] = None

        self.output_bundle_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=ClientRtcConfigModel,
            client_session_delegate_class=RtcClientSessionDelegate,
        )

    def prepare_rtc_definitions(self):
        self.rtc_streamer_factory = RtcStream(
            session_id=None,
            expected_layout="mono",
            input_sample_rate=16000,
            output_sample_rate=24000,
            output_frame_size=480,
            fps=30,
            stream_start_delay=0.5,
        )
        self.rtc_streamer_factory.client_handler_delegate = self.handler_delegate

        audio_output_definition = DataBundleDefinition()
        audio_output_definition.add_entry(DataBundleEntry.create_audio_entry(
            "mic_audio",
            1,
            16000,
        ))
        audio_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.AUDIO] = audio_output_definition

        video_output_definition = DataBundleDefinition()
        video_output_definition.add_entry(DataBundleEntry.create_framed_entry(
            "camera_video",
            [VariableSize(), VariableSize(), VariableSize(), 3],
            0,
            30
        ))
        video_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.VIDEO] = video_output_definition

        text_output_definition = DataBundleDefinition()
        text_output_definition.add_entry(DataBundleEntry.create_text_entry(
            "human_text",
        ))
        text_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.TEXT] = text_output_definition

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        self.engine_config = engine_config
        self.handler_config = cast(ClientRtcConfigModel, handler_config)
        self.prepare_rtc_definitions()

    def setup_rtc_ui(self, ui, parent_block, fastapi: FastAPI, avatar_config):
        turn_entity = RTCProvider().prepare_rtc_configuration(self.handler_config.turn_config)
        if turn_entity is None:
            turn_entity = RTCProvider().prepare_rtc_configuration(self.engine_config.turn_config)

        webrtc = Stream(
            modality="audio-video",
            mode="send-receive",
            time_limit=self.handler_config.connection_ttl,
            rtc_configuration=turn_entity.rtc_configuration if turn_entity is not None else None,
            handler=self.rtc_streamer_factory,
            concurrency_limit=self.handler_config.concurrent_limit,
        )
        webrtc.mount(fastapi)

        @fastapi.get('/soundtech/initconfig')
        async def init_config():
            config = {
                "avatar_config": avatar_config,
                "rtc_configuration": turn_entity.rtc_configuration if turn_entity is not None else None,
            }
            return JSONResponse(status_code=200, content=config)

        @fastapi.get('/session/{session_id}/input')
        async def input_to_session(session_id: str, text: str):
            session_delegate = self.handler_delegate.find_session_delegate(session_id)
            if session_delegate is None:
                msg = f"未找到会话 {session_id}。"
                logger.error(msg)
                return JSONResponse(status_code=404, content={"error": msg})
            try:
                # 直接向 TTS 输入 AVATAR_TEXT，不经过大模型问答
                definition = DataBundleDefinition()
                definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
                data_bundle = DataBundle(definition)
                data_bundle.set_main_data(text)
                data_bundle.add_meta('speech_id', str(uuid4()))
                data_bundle.add_meta('avatar_text_end', True)

                chat_data = ChatData(
                    source="client",
                    type=ChatDataType.AVATAR_TEXT,
                    data=data_bundle,
                    timestamp=session_delegate.get_timestamp(),
                )
                session_delegate.data_submitter.submit(chat_data)
                return JSONResponse(status_code=200, content={"status": "ok"})
            except Exception as e:
                logger.opt(exception=True).error(f"向会话 {session_id} 直接输入文本失败: {e}")
                return JSONResponse(status_code=500, content={"error": "发送失败"})

        @fastapi.post('/session/{session_id}/input')
        async def input_to_session_post(session_id: str, payload: TextPayload):
            session_delegate = self.handler_delegate.find_session_delegate(session_id)
            if session_delegate is None:
                msg = f"未找到会话 {session_id}。"
                logger.error(msg)
                return JSONResponse(status_code=404, content={"error": msg})
            try:
                definition = DataBundleDefinition()
                definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
                data_bundle = DataBundle(definition)
                data_bundle.set_main_data(payload.text)
                data_bundle.add_meta('speech_id', str(uuid4()))
                data_bundle.add_meta('avatar_text_end', True)

                chat_data = ChatData(
                    source="client",
                    type=ChatDataType.AVATAR_TEXT,
                    data=data_bundle,
                    timestamp=session_delegate.get_timestamp(),
                )
                session_delegate.data_submitter.submit(chat_data)
                return JSONResponse(status_code=200, content={"status": "ok"})
            except Exception as e:
                logger.opt(exception=True).error(f"向会话 {session_id} 直接输入文本失败: {e}")
                return JSONResponse(status_code=500, content={"error": "发送失败"})

        @fastapi.get('/session/{session_id}/answer')
        async def speak_to_session(session_id: str, text: str):
            session_delegate = self.handler_delegate.find_session_delegate(session_id)
            if session_delegate is None:
                msg = f"未找到会话 {session_id}。"
                logger.error(msg)
                return JSONResponse(status_code=404, content={"error": msg})
            try:
                # 将文本注入到会话文本通道，触发后续合成流程
                session_delegate.put_data(EngineChannelType.TEXT, text, loopback=True)
                return JSONResponse(status_code=200, content={"status": "ok"})
            except Exception as e:
                logger.opt(exception=True).error(f"向会话 {session_id} 发送文本失败: {e}")
                return JSONResponse(status_code=500, content={"error": "发送失败"})

        @fastapi.post('/session/{session_id}/answer')
        async def speak_to_session_post(session_id: str, payload: TextPayload):
            session_delegate = self.handler_delegate.find_session_delegate(session_id)
            if session_delegate is None:
                msg = f"未找到会话 {session_id}。"
                logger.error(msg)
                return JSONResponse(status_code=404, content={"error": msg})
            try:
                session_delegate.put_data(EngineChannelType.TEXT, payload.text, loopback=True)
                return JSONResponse(status_code=200, content={"status": "ok"})
            except Exception as e:
                logger.opt(exception=True).error(f"向会话 {session_id} 发送文本失败: {e}")
                return JSONResponse(status_code=500, content={"error": "发送失败"})

        @fastapi.get('/manage/sessions')
        async def list_sessions():
            try:
                engine = self.handler_delegate.engine_ref()
                if engine is None:
                    return JSONResponse(status_code=500, content={"error": "引擎不可用"})
                sessions_info = []
                now_monotonic = time.monotonic()
                now_wall = time.time()
                for sid, chat_session in engine.sessions.items():
                    start_mono = getattr(chat_session.session_context, 'input_start_time', -1.0)
                    uptime_sec = 0.0
                    created_at_iso = None
                    if start_mono and start_mono > 0:
                        uptime_sec = max(0.0, now_monotonic - start_mono)
                        created_at_epoch = now_wall - uptime_sec
                        created_at_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(created_at_epoch))
                    sessions_info.append({
                        'id': sid,
                        'created_at_iso': created_at_iso,
                        'uptime_seconds': round(uptime_sec, 3),
                    })
                sessions_info.sort(key=lambda x: (x['created_at_iso'] or ''), reverse=True)
                return JSONResponse(status_code=200, content={"sessions": sessions_info})
            except Exception as e:
                logger.opt(exception=True).error(f"获取会话列表失败: {e}")
                return JSONResponse(status_code=500, content={"error": "获取失败"})

        @fastapi.get('/session/{session_id}/history')
        async def session_history(session_id: str, page: int = 1, page_size: int = 20):
            try:
                engine = self.handler_delegate.engine_ref()
                if engine is None:
                    return JSONResponse(status_code=500, content={"error": "引擎不可用"})
                chat_session = engine.sessions.get(session_id)
                if chat_session is None:
                    msg = f"未找到会话 {session_id}。"
                    logger.error(msg)
                    return JSONResponse(status_code=404, content={"error": msg})
                history_list = None
                for _name, record in chat_session.handlers.items():
                    ctx = getattr(record.env, 'context', None)
                    hist = getattr(ctx, 'history', None) if ctx is not None else None
                    msg_hist = getattr(hist, 'message_history', None) if hist is not None else None
                    if isinstance(msg_hist, list):
                        history_list = msg_hist
                        break
                if history_list is None:
                    return JSONResponse(status_code=200, content={"items": [], "total": 0, "page": page, "page_size": page_size})
                total = len(history_list)
                start = max(0, (page - 1) * page_size)
                end = min(total, start + page_size)
                items = []
                for m in history_list[start:end]:
                    items.append({
                        'role': getattr(m, 'role', None),
                        'content': getattr(m, 'content', ''),
                        'timestamp': getattr(m, 'timestamp', None),
                    })
                return JSONResponse(status_code=200, content={"items": items, "total": total, "page": page, "page_size": page_size})
            except Exception as e:
                logger.opt(exception=True).error(f"获取会话 {session_id} 历史失败: {e}")
                return JSONResponse(status_code=500, content={"error": "获取失败"})

        frontend_path = Path(DirectoryInfo.get_src_dir() + '/handlers/client/rtc_client/frontend')
        if frontend_path.exists():
            logger.info(f"从 {frontend_path} 提供前端服务")
            fastapi.mount('/dev', StaticFiles(directory=frontend_path), name="static")
            fastapi.add_route('/', RedirectResponse(url='/dev/index.html'))
            fastapi.add_route('/manage', RedirectResponse(url='/dev/manage.html'))
        else:
            logger.warning(f"前端目录 {frontend_path} 不存在")
            fastapi.add_route('/', RedirectResponse(url='/gradio'))

        if parent_block is None:
            parent_block = ui
        with ui:
            with parent_block:
                gradio.components.HTML(
                    """
                    <h1 id="openavatarchat">
                       The Gradio page is no longer available. Please use the openavatarchat-webui submodule instead.
                    </h1>
                    """,
                    visible=True
                )

    def on_setup_app(self, app: FastAPI, ui: gradio.blocks.Block, parent_block: Optional[gradio.blocks.Block] = None):
        avatar_config = {}
        self.setup_rtc_ui(ui, parent_block, app, avatar_config)

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        if not isinstance(handler_config, ClientRtcConfigModel):
            handler_config = ClientRtcConfigModel()
        context = ClientRtcContext(session_context.session_info.session_id)
        context.config = handler_config
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def on_setup_session_delegate(self, session_context: SessionContext, handler_context: HandlerContext,
                                  session_delegate: ClientSessionDelegate):
        handler_context = cast(ClientRtcContext, handler_context)
        session_delegate = cast(RtcClientSessionDelegate, session_delegate)

        session_delegate.timestamp_generator = session_context.get_timestamp
        session_delegate.data_submitter = handler_context.data_submitter
        session_delegate.input_data_definitions = self.output_bundle_definitions
        session_delegate.shared_states = session_context.shared_states

        handler_context.client_session_delegate = session_delegate

    def create_handler_detail(self, _session_context, _handler_context):
        inputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO
            ),
            ChatDataType.AVATAR_VIDEO: HandlerDataInfo(
                type=ChatDataType.AVATAR_VIDEO
            ),
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT
            ),
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT
            ),
        }
        outputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO,
                definition=self.output_bundle_definitions[EngineChannelType.AUDIO]
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
                definition=self.output_bundle_definitions[EngineChannelType.VIDEO]
            ),
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=self.output_bundle_definitions[EngineChannelType.TEXT]
            ),
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs
        )

    def get_handler_detail(self, session_context: SessionContext, context: HandlerContext) -> HandlerDetail:
        return self.create_handler_detail(session_context, context)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(ClientRtcContext, context)
        if context.client_session_delegate is None:
            return
        data_queue = context.client_session_delegate.output_queues.get(inputs.type.channel_type)
        if data_queue is not None:
            data_queue.put_nowait(inputs)

    def destroy_context(self, context: HandlerContext):
        pass
