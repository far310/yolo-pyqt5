# setup.py
from cx_Freeze import setup, Executable

build_exe_options = {"packages": ["os"], "include_files": ["model/", "img/", "out"]}

setup(
    name="Object Detection",
    version="0.1",
    description="Object Detection",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", target_name="objectDetection")],
)
