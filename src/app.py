import os
import shutil
import asyncio
import uvicorn
import argparse
import gradio as gr
from loguru import logger
from fastapi import FastAPI

from chat_engine.chat_engine import ChatEngine
from engine_utils.directory_info import DirectoryInfo
from service.service_utils.logger_utils import config_loggers
from service.service_utils.ssl_helpers import create_ssl_context
from service.service_utils.service_config_loader import load_configs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="service host address")
    parser.add_argument("--port", type=int, help="service host port")
    parser.add_argument("--config", type=str, default="config\config.yaml", help="config file to use")
    return parser.parse_args()

class OpenAvatarChatWebServer(uvicorn.Server):

    def __init__(self, chat_engine: ChatEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_engine = chat_engine
    
    async def shutdown(self, sockets=None):
        logger.info("开始常规关闭流程")
        self.chat_engine.shutdown()
        await super().shutdown(sockets)     

def setup_app():
    app = FastAPI()

    css = """

    .app {
        @media screen and (max-width: 768px) {
            padding: 8px !important;
        }
    }
    footer {
        display: none !important;
    }
    """
    with gr.Blocks(css=css) as gradio_block:
        with gr.Column():
            with gr.Group() as rtc_container:
                pass
    return app, gradio_block, rtc_container


def main():
    args = parse_args()
    logger_config, service_config, engine_config = load_configs(args)

    if not os.path.isabs(engine_config.model_root):
        os.environ['MODELSCOPE_CACHE'] = os.path.join(DirectoryInfo.get_project_dir(), engine_config.model_root.replace('models', ''))

    config_loggers(logger_config)
    chat_engine = ChatEngine()
    app, ui, parent_block = setup_app()

    chat_engine.initialize(engine_config, app=app, ui=ui, parent_block=parent_block)

    ssl_context = create_ssl_context(args, service_config)

    uvicorn_config = uvicorn.Config(
        app,
        host=service_config.host,
        port=service_config.port,
        access_log=False,
        **ssl_context,
    )
    server = OpenAvatarChatWebServer(chat_engine, uvicorn_config)
    try:
        server.run()
    except KeyboardInterrupt:
        try:
            server.shutdown()
        except Exception as e:
            logger.error(f"关闭服务时出错：{e}")
        # 清除项目内 __pycache__ 缓存
        project_dir = DirectoryInfo.get_project_dir()
        venv_dir = os.path.join(project_dir, '.venv')
        for root, dirs, files in os.walk(project_dir):
            if root.startswith(venv_dir):
                continue
            if '__pycache__' in dirs:
                pycache_dir = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(pycache_dir)
                    # logger.info(f"已删除缓存目录: {pycache_dir}")
                except Exception as e:
                    logger.warning(f"删除缓存目录 {pycache_dir} 失败：{e}")
        logger.info("服务已退出。")
    except asyncio.CancelledError:
        logger.info("事件循环取消，服务正常退出。")


if __name__ == "__main__":
    main()
