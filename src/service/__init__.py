import os
import subprocess
from loguru import logger
from threading import Thread
from engine_utils.directory_info import DirectoryInfo

def frpc_server():
    logger.info("启动 FRP 内网穿透服务")
    project_dir = DirectoryInfo.get_project_dir()
    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "frpc.log")
    try:
        with open(log_path, "ab+") as log_file:
            subprocess.Popen(["ssl_certs\\frpc.exe", "-c", "ssl_certs\\frpc.toml"], stdout=log_file, stderr=log_file)
    except Exception as e:
        logger.error(f"启动 FRP 内网穿透服务失败: {e}")
    
def ollama_server():
    logger.info("启动 Ollama 服务")
    project_dir = DirectoryInfo.get_project_dir()
    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "ollama.log")
    try:
        with open(log_path, "ab+") as log_file:
            subprocess.Popen(["ollama", "serve"], stdout=log_file, stderr=log_file)
    except Exception as e:
        logger.error(f"启动 Ollama 服务失败: {e}")

frp_thread = Thread(target=frpc_server, daemon=True)
frp_thread.start()

ollama_thread = Thread(target=ollama_server, daemon=True)
ollama_thread.start()
