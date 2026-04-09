from pathlib import Path
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


THIS_DIR = Path(__file__).resolve().parent
UOPC_INSTALL_ROOT = Path(
    os.environ.get(
        "UOPC_INSTALL_ROOT",
        "/apps/qiongzhu/rocm-overlap-policy/install-uopc-four-bucket",
    )
)
UOPC_INCLUDE_DIR = Path(
    os.environ.get(
        "UOPC_INCLUDE_DIR",
        "/apps/qiongzhu/rocm-overlap-policy/include",
    )
)
UOPC_LIB_DIR = UOPC_INSTALL_ROOT / "lib"
ROCM_PATH = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
ROCM_INCLUDE_DIR = ROCM_PATH / "include"

extension = CppExtension(
    name="torch_uopc_patch._C",
    sources=[str(THIS_DIR / "csrc" / "process_group_uopc.cpp")],
    include_dirs=[str(UOPC_INCLUDE_DIR), str(ROCM_INCLUDE_DIR)],
    library_dirs=[str(UOPC_LIB_DIR)],
    libraries=["uopc"],
    extra_compile_args={
        "cxx": [
            "-O2",
            "-std=c++17",
            "-fvisibility=hidden",
            "-fdiagnostics-color=always",
        ]
    },
    extra_link_args=[f"-Wl,-rpath,{UOPC_LIB_DIR}"],
)

setup(
    name="torch-uopc-patch",
    version="0.1.0",
    description="Standalone UOPC ProcessGroup backend for installed PyTorch",
    packages=find_packages(),
    py_modules=["sitecustomize"],
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
