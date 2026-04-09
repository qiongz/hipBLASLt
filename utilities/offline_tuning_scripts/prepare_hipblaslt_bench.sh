#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: prepare_hipblaslt_bench.sh [options]

Build a client-enabled hipblaslt-bench from the current feat/uopc branch.

Options:
  --build-dir <path>        Build directory. Default: <repo>/build-uopc-clients
  --install-prefix <path>   Optional install prefix. Default: <build-dir>/install
  --gpu-targets <value>     GPU targets passed to CMake. Default: gfx942
  --jobs <value>            Parallel build jobs. Default: nproc
  --uopc-dir <path>         Path to uopc package config dir.
                            Default: /apps/qiongzhu/rocm-overlap-policy/install-uopc-four-bucket/lib/cmake/uopc
  --hip-dir <path>          Path to hip package config dir.
                            Default: auto-detect from /opt/TheRock
  --hipblas-common-dir <path>
                            Path to hipblas-common package config dir.
                            Default: auto-detect from /opt/TheRock
  --llvm-bin <path>         Path to ROCm LLVM bin dir that contains amdclang/amdclang++
                            Default: /opt/rocm/core-7.11/lib/llvm/bin
  --amddevicelibs-dir <path>
                            Path to AMDDeviceLibs CMake package dir.
                            Default: /opt/TheRock/7.10.0a20251120/lib/cmake/AMDDeviceLibs
  --clean                   Remove the build directory before configure
  --install                 Run cmake --install after build
  --help                    Show this message
EOF
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)

pick_first_dir() {
    local candidate
    for candidate in "$@"; do
        if [[ -d "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

join_existing_prefixes() {
    local result=""
    local candidate
    for candidate in "$@"; do
        if [[ -d "${candidate}" ]]; then
            if [[ -n "${result}" ]]; then
                result="${result};${candidate}"
            else
                result="${candidate}"
            fi
        fi
    done
    printf '%s\n' "${result}"
}

build_dir="${repo_root}/build-uopc-clients"
install_prefix=""
gpu_targets="${GPU_TARGETS:-gfx942}"
jobs="${BUILD_JOBS:-$(nproc)}"
uopc_dir="${UOPC_DIR:-/apps/qiongzhu/rocm-overlap-policy/install-uopc-four-bucket/lib/cmake/uopc}"
hip_dir="${HIP_DIR:-$(pick_first_dir \
    /opt/rocm/lib/cmake/hip \
    /opt/rocm-7.1.0/lib/cmake/hip \
    /opt/TheRock/7.10.0a20251120/lib/cmake/hip \
    /opt/TheRock/7.11.0a20251120/lib/cmake/hip)}"
hipblas_common_dir="${HIPBLAS_COMMON_DIR:-$(pick_first_dir \
    /opt/rocm/lib/cmake/hipblas-common \
    /opt/rocm-7.1.0/lib/cmake/hipblas-common \
    /opt/TheRock/7.10.0a20251120/lib/cmake/hipblas-common \
    /opt/TheRock/7.11.0a20251120/lib/cmake/hipblas-common)}"
llvm_bin="${ROCM_LLVM_BIN:-$(pick_first_dir \
    /opt/rocm-7.1.0/lib/llvm/bin \
    /opt/rocm/lib/llvm/bin \
    /opt/rocm/core-7.11/lib/llvm/bin \
    /opt/TheRock/7.10.0a20251120/lib/llvm/bin)}"
amddevicelibs_dir="${AMDDeviceLibs_DIR:-$(pick_first_dir \
    /opt/rocm/lib/cmake/AMDDeviceLibs \
    /opt/rocm-7.1.0/lib/llvm/lib/cmake/AMDDeviceLibs \
    /opt/rocm-7.1.0/lib/cmake/AMDDeviceLibs \
    /opt/TheRock/7.10.0a20251120/lib/cmake/AMDDeviceLibs \
    /opt/TheRock/7.10.0a20251120/lib/llvm/lib/cmake/AMDDeviceLibs)}"
hip_root="$(cd "${hip_dir}/../../.." && pwd)"
cmake_prefix_path="${CMAKE_PREFIX_PATH:-$(join_existing_prefixes \
    "${hip_root}" \
    /opt/rocm \
    /opt/rocm-7.1.0 \
    /opt/rocm/core-7.11 \
    /opt/rocm/core-7 \
    /opt/rocm/core \
    /opt/TheRock/7.10.0a20251120)}"
blas_libraries="${BLAS_LIBRARIES:-/usr/lib/x86_64-linux-gnu/libblas.so.3}"
lapack_libraries="${LAPACK_LIBRARIES:-/usr/lib/x86_64-linux-gnu/liblapack.so.3}"
do_clean=0
do_install=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            build_dir="$2"
            shift 2
            ;;
        --install-prefix)
            install_prefix="$2"
            shift 2
            ;;
        --gpu-targets)
            gpu_targets="$2"
            shift 2
            ;;
        --jobs)
            jobs="$2"
            shift 2
            ;;
        --uopc-dir)
            uopc_dir="$2"
            shift 2
            ;;
        --hip-dir)
            hip_dir="$2"
            shift 2
            ;;
        --hipblas-common-dir)
            hipblas_common_dir="$2"
            shift 2
            ;;
        --llvm-bin)
            llvm_bin="$2"
            shift 2
            ;;
        --amddevicelibs-dir)
            amddevicelibs_dir="$2"
            shift 2
            ;;
        --clean)
            do_clean=1
            shift
            ;;
        --install)
            do_install=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${install_prefix}" ]]; then
    install_prefix="${build_dir}/install"
fi

if [[ -z "${hip_dir}" || -z "${hipblas_common_dir}" || -z "${llvm_bin}" || -z "${amddevicelibs_dir}" ]]; then
    echo "Failed to auto-detect required ROCm package paths." >&2
    echo "hip_dir=${hip_dir:-<missing>}" >&2
    echo "hipblas_common_dir=${hipblas_common_dir:-<missing>}" >&2
    echo "llvm_bin=${llvm_bin:-<missing>}" >&2
    echo "amddevicelibs_dir=${amddevicelibs_dir:-<missing>}" >&2
    exit 1
fi

if [[ ${do_clean} -eq 1 ]]; then
    rm -rf "${build_dir}"
fi

cxx_compiler="${llvm_bin}/amdclang++"
c_compiler="${llvm_bin}/amdclang"
asm_compiler="${llvm_bin}/amdclang"

if [[ ! -x "${cxx_compiler}" || ! -x "${c_compiler}" || ! -x "${asm_compiler}" ]]; then
    echo "Missing ROCm LLVM compilers under ${llvm_bin}" >&2
    exit 1
fi

export HIP_PLATFORM="${HIP_PLATFORM:-amd}"
export HIP_PATH="${HIP_PATH:-${hip_root}}"

cmake --preset hipblaslt-clients -S "${repo_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
    -DCMAKE_CXX_COMPILER="${cxx_compiler}" \
    -DCMAKE_C_COMPILER="${c_compiler}" \
    -DCMAKE_ASM_COMPILER="${asm_compiler}" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PREFIX_PATH="${cmake_prefix_path}" \
    -DGPU_TARGETS="${gpu_targets}" \
    -DHIPBLASLT_ENABLE_UOPC=ON \
    -Duopc_DIR="${uopc_dir}" \
    -Dhip_DIR="${hip_dir}" \
    -Dhipblas-common_DIR="${hipblas_common_dir}" \
    -DAMDDeviceLibs_DIR="${amddevicelibs_dir}" \
    -DBLA_VENDOR=Generic \
    -DBLAS_LIBRARIES="${blas_libraries}" \
    -DLAPACK_LIBRARIES="${lapack_libraries}" \
    -DCMAKE_DISABLE_FIND_PACKAGE_rocm_smi=ON \
    -DHIPBLASLT_ENABLE_ROCM_SMI=OFF \
    -DHIPBLASLT_ENABLE_BLIS=OFF \
    -DHIPBLASLT_BUILD_TESTING=OFF \
    -DBUILD_TESTING=OFF \
    -DHIPBLASLT_ENABLE_SAMPLES=OFF

cmake --build "${build_dir}" --target hipblaslt-bench --parallel "${jobs}"

if [[ ${do_install} -eq 1 ]]; then
    cmake --install "${build_dir}"
fi

bench_path="${build_dir}/clients/staging/hipblaslt-bench"
if [[ ! -x "${bench_path}" ]]; then
    bench_path="${build_dir}/clients/hipblaslt-bench"
fi

if [[ ! -x "${bench_path}" ]]; then
    echo "Failed to locate hipblaslt-bench under ${build_dir}" >&2
    exit 1
fi

bench_library_dir="${build_dir}/library"
tensile_libpath="${build_dir}/Tensile/library"
if ! compgen -G "${tensile_libpath}/TensileLibrary_lazy_*" > /dev/null; then
    if [[ -d "/opt/rocm/lib/hipblaslt/library" ]]; then
        tensile_libpath="/opt/rocm/lib/hipblaslt/library"
    else
        tensile_libpath=""
    fi
fi

echo "hipblaslt-bench ready: ${bench_path}"
echo "export HIPBLASLT_BENCH_PATH=\"${bench_path}\""
if [[ -d "${bench_library_dir}" ]]; then
    echo "export HIPBLASLT_LIBRARY_DIR=\"${bench_library_dir}\""
    echo "export LD_LIBRARY_PATH=\"${bench_library_dir}:\${LD_LIBRARY_PATH}\""
fi
if [[ -n "${tensile_libpath}" ]]; then
    echo "export HIPBLASLT_TENSILE_LIBPATH=\"${tensile_libpath}\""
fi
