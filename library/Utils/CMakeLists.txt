include_directories(include)
set(UTILS_NAME utils)
set(UTILS_SOURCES
        src/IO.cpp
        src/DataInitialization.cpp
        src/LegoDataLoader.cpp)

add_library(${UTILS_NAME} ${UTILS_SOURCES})
target_include_directories(${UTILS_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(${UTILS_NAME} eigenI lodepng mnist)
set_target_properties(${UTILS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)