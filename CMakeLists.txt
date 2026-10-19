cmake_minimum_required(VERSION 3.24)

project(bpe LANGUAGES CXX)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_CUDNN "Build the cuDNN attention path" OFF)

# ---------- Optional CUDA (never on macOS) ----------
set(USE_CUDA OFF)
if(NOT APPLE)
    # Must be set BEFORE the CUDA language is enabled (enabling it fills in a default)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(USE_CUDA ON)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)
    else()
        message(WARNING "No CUDA compiler found: building CPU-only. "
                        "Set -DCMAKE_CUDA_COMPILER=<path to nvcc.exe> and delete the CMake cache.")
    endif()
endif()
message(STATUS "USE_CUDA=${USE_CUDA}  USE_CUDNN=${USE_CUDNN}  CUDA archs=${CMAKE_CUDA_ARCHITECTURES}")

# ---------- Sources ----------
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
if(USE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu"
    )
    if(NOT USE_CUDNN)
        list(FILTER CUDA_SOURCES EXCLUDE REGEX "cudnn_attention")
    endif()
    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

add_executable(main ${SOURCES})

target_include_directories(main PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Warnings
if(MSVC)
    target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /bigobj>)
else()
    target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
endif()

# ---------- CUDA / cuDNN ----------
# ---------- CUDA / cuDNN ----------
if(USE_CUDA)
    target_compile_definitions(main PRIVATE USE_CUDA=1)

    if(MSVC)
        target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    else()
        target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    endif()

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(main PRIVATE CUDA::cudart CUDA::cublas)

    if(USE_CUDNN)
        target_compile_definitions(main PRIVATE USE_CUDNN=1)

        include(FetchContent)
        FetchContent_Declare(
            cudnn_frontend
            GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
            GIT_TAG        v1.16.0
            SOURCE_SUBDIR  _no_cmake_here
        )
        FetchContent_MakeAvailable(cudnn_frontend)
        target_include_directories(main SYSTEM PRIVATE
            "${cudnn_frontend_SOURCE_DIR}/include")

        if(WIN32)
            set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.26" CACHE PATH "cuDNN install root")
            set(CUDNN_VER  "13.4" CACHE STRING "cuDNN CUDA subfolder")
            target_include_directories(main SYSTEM PRIVATE "${CUDNN_ROOT}/include/${CUDNN_VER}")
            target_link_directories(main PRIVATE "${CUDNN_ROOT}/lib/${CUDNN_VER}/x64")
            target_link_libraries(main PRIVATE cudnn)
            add_custom_command(TARGET main POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CUDNN_ROOT}/bin/${CUDNN_VER}/x64" "$<TARGET_FILE_DIR:main>"
                COMMENT "Copying cuDNN DLLs")
            else()
                target_link_libraries(main PRIVATE cudnn)
        endif()
    endif()
endif()

# ---------- Platform dependencies ----------
if(APPLE)
    set(BREW_PREFIX /opt/homebrew/opt)
    target_include_directories(main SYSTEM PRIVATE
        ${BREW_PREFIX}/pcre2/include
        ${BREW_PREFIX}/apache-arrow/include)
    target_link_directories(main PRIVATE
        ${BREW_PREFIX}/pcre2/lib
        ${BREW_PREFIX}/apache-arrow/lib)
    target_link_libraries(main PRIVATE pcre2-8 parquet arrow)

elseif(WIN32)
    set(VCPKG_PREFIX "C:/vcpkg/installed/x64-windows")

    function(find_vcpkg_dep var header lib)
        find_path(${var}_INCLUDE_DIR NAMES ${header}
            PATHS "${VCPKG_PREFIX}/include" NO_DEFAULT_PATH REQUIRED)
        find_library(${var}_LIBRARY NAMES ${lib}
            PATHS "${VCPKG_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
    endfunction()

    find_vcpkg_dep(PCRE2   pcre2.h              pcre2-8)
    find_vcpkg_dep(ARROW   arrow/api.h          arrow)
    find_vcpkg_dep(PARQUET parquet/api/reader.h parquet)

    target_include_directories(main SYSTEM PRIVATE
        ${PCRE2_INCLUDE_DIR} ${ARROW_INCLUDE_DIR} ${PARQUET_INCLUDE_DIR})
    target_link_libraries(main PRIVATE
        ${PCRE2_LIBRARY} ${PARQUET_LIBRARY} ${ARROW_LIBRARY})

    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${VCPKG_PREFIX}/bin" "$<TARGET_FILE_DIR:main>"
        COMMENT "Copying vcpkg runtime DLLs")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()