pybind11_add_module(libdl bindings.cpp)
target_link_libraries(libdl PUBLIC session utils pybind11I)

#add_custom_command(
#        ...
#        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH="/path/to/modules" python the_script.py
#)
#include(GNUInstallDirs)
#
#install(TARGETS libdl
#        COMPONENT python
#        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
#        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
#
#message(${CMAKE_INSTALL_LIBDIR})