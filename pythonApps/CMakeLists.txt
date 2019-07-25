cmake_minimum_required(VERSION 3.12)

#configure_file(test.py ${CMAKE_CURRENT_BINARY_DIR}/library/bindings/test.py COPYONLY)

enable_testing()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME MatmulPython COMMAND Python3::Interpreter matmul.py WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
#    add_custom_target(tt MatmulPython COMMAND Python3::Interpreter test.py WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()

#find_package(PythonLibs REQUIRED)
#add_executable(pythontest test.py)
#target_link_libraries(pythontest libdl)

#add_custom_target(runner COMMAND python test.py)
#
#add_custom_target(run
#        COMMAND cmake "-E" "env"  "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/library/bindings/ DYLD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/library/bindings/" "python" "./test.py"
#                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}    )
#
#
