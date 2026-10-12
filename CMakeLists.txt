cmake_minimum_required(VERSION 3.20)

project(bpe LANGUAGES CXX)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find all .cpp files under src/, including subdirectories
file(GLOB_RECURSE SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
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

# Homebrew dependencies
target_include_directories(main SYSTEM PRIVATE
    /opt/homebrew/opt/pcre2/include
    /opt/homebrew/opt/apache-arrow/include
)

target_link_directories(main PRIVATE
    /opt/homebrew/opt/pcre2/lib
    /opt/homebrew/opt/apache-arrow/lib
)

target_link_libraries(main PRIVATE
    pcre2-8
    parquet
    arrow
)