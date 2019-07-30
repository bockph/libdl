
set(bindingsName libdl)

pybind11_add_module(${bindingsName} bindings.cpp)
target_link_libraries(${bindingsName} PUBLIC session utils )
set_target_properties(${bindingsName} PROPERTIES POSITION_INDEPENDENT_CODE ON)
