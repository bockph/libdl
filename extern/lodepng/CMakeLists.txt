include_directories(include)
add_library(lodepng src/lodepng.cpp)
target_include_directories(lodepng INTERFACE  ${CMAKE_CURRENT_LIST_DIR}/include)