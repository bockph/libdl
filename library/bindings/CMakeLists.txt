

pybind11_add_module(libdl bindings.cpp)
target_link_libraries(libdl PUBLIC session utils )
