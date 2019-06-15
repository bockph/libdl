include_directories(include)
include_directories(Operations/include)
set(GRAPH_NAME graph)
set(GRAPH_SOURCES
         src/Node.cpp
         src/Operation.cpp
         src/Placeholder.cpp
         src/Graph.cpp
         src/Session.cpp
         Operations/src/SUM.cpp
        Operations/src/MUL.cpp
        Operations/src/Sigmoid.cpp
        Operations/src/MSE.cpp
        src/Weight.cpp
        src/Bias.cpp
        src/Filter.cpp
        Operations/src/ConvolveFilter.cpp
        )

add_library(${GRAPH_NAME}  ${GRAPH_SOURCES})
target_include_directories(${GRAPH_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/Operations/include)
target_link_libraries(${GRAPH_NAME} Eigen3::Eigen)
set_target_properties(${GRAPH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_subdirectory(Operations/tests)
add_subdirectory(tests)
