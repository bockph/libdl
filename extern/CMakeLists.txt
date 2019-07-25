find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/eigen/CMakeLists.txt")
    message(FATAL_ERROR "Problems with submodule Eigen3.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Catch2/CMakeLists.txt")
    message(FATAL_ERROR "Problems with submodule CATCH2.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    message(FATAL_ERROR "Problems with submodule pybind.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/spdlog/CMakeLists.txt")
    message(FATAL_ERROR "Problems with submodule  spdlog.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/mnist/MNISTConfig.cmake")
    message(FATAL_ERROR "Problems with submodule  mnist.")
endif()
find_package(OpenMP)

add_library(eigenI INTERFACE)
target_include_directories(eigenI INTERFACE ${PROJECT_SOURCE_DIR}/extern/eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(eigenI INTERFACE  OpenMP::OpenMP_CXX)
endif()
add_library(catch2I INTERFACE)
target_include_directories(catch2I INTERFACE ${PROJECT_SOURCE_DIR}/extern/Catch2/single_include)
#target_include_directories(catch2I INTERFACE ${PROJECT_SOURCE_DIR}/extern/Catch2/internal)
#
#target_sources(catch2I INTERFACE ${PROJECT_SOURCE_DIR}/extern/Catch2/include/internal)
add_library(spdlogI INTERFACE)
target_include_directories(spdlogI INTERFACE ${PROJECT_SOURCE_DIR}/extern/spdlog/include)
add_subdirectory(lodepng)


add_library(mnist INTERFACE)
target_include_directories(mnist INTERFACE ${PROJECT_SOURCE_DIR}/extern/mnist/include)





#
#set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen/Eigen CACHE PATH "eigen include directory")
#set(install {CMAKE_CURRENT_SOURCE_DIR}/extern/eigenInstall)
#install(DIRECTORY ${EIGEN_INCLUDE_DIR} DESTINATION  ${install})
#add_library(eigenI INTERFACE)
#target_include_directories(eigenI INTERFACE $<BUILD_INTERFACE:${FOO_INCLUDE_DIR}>
#        $<INSTALL_INTERFACE:${install}>)
#install(TARGETS eigenI EXPORT my_export_set DESTINATION ${install})
##add_subdirectory(eigen)
#add_subdirectory(Catch2)
#add_subdirectory(pybind11)
#add_subdirectory(spdlog)

#//https://foonathan.net/blog/2016/07/07/cmake-dependency-handling.html <-- do this, see E-mail Tobias