cmake_minimum_required(VERSION 3.24)

# CUDA architecture of this build: `native` asks the installed GPU at configure time; a list such as 86 or 86-real
# overrides it (-DCMAKE_CUDA_ARCHITECTURES=...).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(strata_q36 VERSION 0.1.49.2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ---- ggml: the CPU backend and the base library, built static from the vendored llama.cpp tree.  The CPU computes the
# experts that do not fit in VRAM with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so a Q4_K weight means
# exactly what it means in llama.cpp.  GGML_NATIVE builds ggml-cpu for the build host.
set(STRATA_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH "llama.cpp tree to take ggml from")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
set_target_properties(ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}" STRATA_Q6K_EXPERTS=1)

# Tuning hooks: extra flags for this project's own C++ and CUDA sources (not for ggml, which has its own GGML_* options).
set(STRATA_CXX_FLAGS "" CACHE STRING "extra C++ flags for Strata's own sources, e.g. -march=native")
set(STRATA_CUDA_FLAGS "" CACHE STRING "extra nvcc flags for Strata's own sources")
separate_arguments(_strata_cxx_flags NATIVE_COMMAND "${STRATA_CXX_FLAGS}")
separate_arguments(_strata_cuda_flags NATIVE_COMMAND "${STRATA_CUDA_FLAGS}")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${_strata_cxx_flags}>" "$<$<COMPILE_LANGUAGE:CUDA>:${_strata_cuda_flags}>")

set(_ggml_include ${STRATA_GGML_DIR}/ggml/include)
set(_ggml_cuda ${STRATA_GGML_DIR}/ggml/src/ggml-cuda)
set(_strata_include ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- CPU expert kernels.  Strata's own AVX2 / AVX-512 rows are compiled per file and picked at run time.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
check_cxx_source_compiles([=[
__attribute__((target("avxvnni")))
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set(_strata_avxvnni STRATA_AVXVNNI=1)
else()
  set(_strata_avxvnni STRATA_AVXVNNI=0)
endif()

add_library(strata_kernels_cpu STATIC
  src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp src/kernels/cpu/expert_layout.cpp src/kernels/cpu/hadamard.cpp
  src/kernels/cpu/native_expert.cpp src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/q2_avx1.cpp
  src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/iq_avx1.cpp
  src/kernels/cpu/kq_avx2.cpp src/kernels/cpu/kq_avx512.cpp
  src/kernels/cpu/nvfp4_avx512.cpp src/kernels/cpu/nvfp4_avx2.cpp
  src/kernels/cpu/tq_cpu.cpp src/kernels/cpu/tq_avx2.cpp src/kernels/cpu/tq_avx512.cpp)
target_include_directories(strata_kernels_cpu PUBLIC ${_strata_include} PRIVATE third_party/ggml ${_ggml_include})
target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base Threads::Threads)
set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/nvfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mavx512vpopcntdq;-mbmi2;-mf16c;-mfma")
set_source_files_properties(src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp src/kernels/cpu/nvfp4_avx2.cpp
  src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mavx512vbmi;-mavx2;-mfma;-mf16c")
set_source_files_properties(src/kernels/cpu/kq_avx512.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx2;-mfma;-mf16c;-mbmi2")
set_source_files_properties(src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS
  "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
set_property(SOURCE src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})

# Bit-exactness of the AVX-512 expert kernels against ggml-cpu: `ninja kq512_parity && ./kq512_parity [--bench]`
add_executable(kq512_parity EXCLUDE_FROM_ALL tests/kq512_parity.cpp)
target_include_directories(kq512_parity PRIVATE ${_strata_include} ${_ggml_include} third_party/ggml)
target_link_libraries(kq512_parity PRIVATE strata_kernels_cpu)

# ---- CUDA kernels of the Qwen3.6 path
add_library(strata_cuda_kernels STATIC
  src/kernels/cuda/dequant_bf16.cu src/kernels/cuda/iq_kernels.cu src/kernels/cuda/native_mmvq.cu
  src/kernels/cuda/native_router.cu src/kernels/cuda/verify_kernels.cu)
target_include_directories(strata_cuda_kernels PUBLIC ${_strata_include} PRIVATE third_party/ggml)
target_compile_definitions(strata_cuda_kernels PRIVATE STRATA_HAVE_DFLASH2=1)
target_compile_options(strata_cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
target_link_libraries(strata_cuda_kernels PUBLIC CUDA::cudart)

# ---- llama.cpp's MMQ prompt kernels (ggml-cuda, MIT).  Q8_0 is compiled into the engine; the K-quants (Q4_K, Q5_K,
# Q6_K, Q5_1) are libstrata_kq, a shared library next to the engine that is opened on first use: the engine loads every
# kernel of its own binary at start (CUDA_MODULE_LOADING=EAGER), and these four types would take their VRAM from every model.
add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/ggml_cuda_host.cu src/prefill/mmq_kq_loader.cpp
  ${_ggml_cuda}/quantize.cu ${_ggml_cuda}/template-instances/mmq-instance-q8_0.cu)
target_include_directories(strata_mmq PRIVATE ${_ggml_include} ${STRATA_GGML_DIR}/ggml/src ${_ggml_cuda} PUBLIC ${_strata_include})
target_compile_definitions(strata_mmq PRIVATE STRATA_KQ_DSO=1 STRATA_MMQ_KQUANTS=1)
target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
target_link_libraries(strata_mmq PUBLIC ggml-base CUDA::cudart CUDA::cuda_driver ${CMAKE_DL_LIBS})

add_library(strata_kq SHARED src/prefill/mmq_kq_lib.cu src/prefill/ggml_cuda_host.cu
  ${_ggml_cuda}/template-instances/mmq-instance-q4_k.cu ${_ggml_cuda}/template-instances/mmq-instance-q5_k.cu
  ${_ggml_cuda}/template-instances/mmq-instance-q5_1.cu ${_ggml_cuda}/template-instances/mmq-instance-q6_k.cu)
target_include_directories(strata_kq PRIVATE ${_ggml_include} ${STRATA_GGML_DIR}/ggml/src ${_ggml_cuda} ${_strata_include})
target_compile_options(strata_kq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
set_target_properties(strata_kq PROPERTIES CUDA_RUNTIME_LIBRARY Static POSITION_INDEPENDENT_CODE ON
                      CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
target_link_libraries(strata_kq PRIVATE ggml-base CUDA::cuda_driver)
add_dependencies(strata_mmq strata_kq)

# ---- the engine
add_library(strata_prefill STATIC src/prefill/gemm.cu)
target_include_directories(strata_prefill PUBLIC ${_strata_include} PRIVATE ${_ggml_include})
target_compile_definitions(strata_prefill PRIVATE STRATA_HAVE_DFLASH2=1 STRATA_NATIVE_EXPERTS=1 STRATA_PREFILL_FUSED=1
                           STRATA_PREFILL_MMQ=1)
target_link_libraries(strata_prefill PUBLIC strata_mmq CUDA::cublas CUDA::cudart)

add_library(strata_qwen35 STATIC src/qwen35/q36_engine.cu)
target_include_directories(strata_qwen35 PUBLIC ${_strata_include} PRIVATE ${_ggml_include})
target_compile_definitions(strata_qwen35 PRIVATE STRATA_HAVE_DFLASH2=1 STRATA_NATIVE_EXPERTS=1 STRATA_PREFILL_FUSED=1
                           STRATA_PREFILL_MMQ=1 STRATA_Q36_CPU_POOL=1)
target_link_libraries(strata_qwen35 PRIVATE strata_prefill strata_cuda_kernels strata_kernels_cpu)

add_executable(strata-q36 src/qwen35/q36_main.cpp)
target_link_libraries(strata-q36 PRIVATE strata_qwen35 CUDA::cudart)
# libstrata_kq.so is looked up beside the executable (/proc/self/exe)
add_dependencies(strata-q36 strata_kq)
set_target_properties(strata_kq PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")

install(TARGETS strata-q36 strata_kq RUNTIME DESTINATION bin LIBRARY DESTINATION bin)
