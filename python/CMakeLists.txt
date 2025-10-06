# Python bindings for NLFormer
find_package(pybind11 REQUIRED)

pybind11_add_module(nlformer_python
    python_bindings.cpp
)

target_link_libraries(nlformer_python PRIVATE nlformer)

# Set properties for Python module
set_target_properties(nlformer_python PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)
