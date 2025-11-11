import sys
from engine_utils.directory_info import DirectoryInfo

project_dir = DirectoryInfo.get_project_dir()
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)