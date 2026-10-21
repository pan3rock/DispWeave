# All production dependencies are bundled. No FetchContent, network, or package manager.
add_library(dw_eigen INTERFACE)
target_include_directories(dw_eigen SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/eigen)
target_compile_definitions(dw_eigen INTERFACE EIGEN_MPL2_ONLY)
add_library(dw_toml INTERFACE)
target_include_directories(dw_toml SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/toml11/include)
add_library(dw_cli INTERFACE)
target_include_directories(dw_cli SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/CLI11/include)
message(STATUS "Bundled dependencies: Eigen 3.4.1, toml11 4.4.0, CLI11 2.4.2")
