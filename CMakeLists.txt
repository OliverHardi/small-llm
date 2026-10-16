cmake_minimum_required(VERSION 3.20)

project(bpe LANGUAGES CXX)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find all .cpp files under src/
file(GLOB_RECURSE SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Find all .cpp files under cuda/
file(GLOB_RECURSE SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cpp"
)

# Executable
add_executable(main ${SOURCES})

# Compiler warnings
target_compile_options(main PRIVATE
    -Wall
    -Wextra
)

# Your headers
target_include_directories(main PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Platform-specific dependencies
if(APPLE)
    # Homebrew (Apple Silicon)
    set(BREW_PREFIX /opt/homebrew/opt)

    target_include_directories(main SYSTEM PRIVATE
        ${BREW_PREFIX}/pcre2/include
        ${BREW_PREFIX}/apache-arrow/include
    )

    target_link_directories(main PRIVATE
        ${BREW_PREFIX}/pcre2/lib
        ${BREW_PREFIX}/apache-arrow/lib
    )

    target_link_libraries(main PRIVATE
        pcre2-8
        parquet
        arrow
    )

elseif(WIN32)
    # vcpkg (MinGW)
    set(VCPKG_PREFIX "C:/vcpkg/installed/x64-mingw-dynamic")

    # Locate a header + library pair under the vcpkg prefix
    function(find_vcpkg_dep var header lib)
        find_path(${var}_INCLUDE_DIR
            NAMES ${header}
            PATHS "${VCPKG_PREFIX}/include"
            NO_DEFAULT_PATH
            REQUIRED
        )
        find_library(${var}_LIBRARY
            NAMES ${lib}
            PATHS "${VCPKG_PREFIX}/lib"
            NO_DEFAULT_PATH
            REQUIRED
        )
    endfunction()

    find_vcpkg_dep(PCRE2   pcre2.h              pcre2-8)
    find_vcpkg_dep(ARROW   arrow/api.h          arrow)
    find_vcpkg_dep(PARQUET parquet/api/reader.h parquet)

    target_include_directories(main SYSTEM PRIVATE
        ${PCRE2_INCLUDE_DIR}
        ${ARROW_INCLUDE_DIR}
        ${PARQUET_INCLUDE_DIR}
    )

    target_link_libraries(main PRIVATE
        ${PCRE2_LIBRARY}
        ${PARQUET_LIBRARY}
        ${ARROW_LIBRARY}
    )

    # Make the vcpkg DLLs findable at runtime
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${VCPKG_PREFIX}/bin" "$<TARGET_FILE_DIR:main>"
        COMMENT "Copying vcpkg runtime DLLs"
    )

else()
    message(FATAL_ERROR "Unsupported platform")
endif()