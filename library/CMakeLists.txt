# ------------ unit testing macro ----------
# ------------
MACRO (LIBDL_TEST NAME)
    SET(MODULES "${ARGN}")
    ADD_EXECUTABLE(test_${NAME} test_${NAME}.cpp ${PROJECT_SOURCE_DIR}/library/test_main.cpp)
    TARGET_LINK_LIBRARIES(test_${NAME} ${MODULES} eigenI spdlogI catch2I )
    ADD_TEST(test_${NAME} test_${NAME})
ENDMACRO (LIBDL_TEST)



add_subdirectory(Utils)

add_subdirectory(ComputationalGraph)

add_subdirectory(NeuralNetwork)

add_subdirectory(bindings)


