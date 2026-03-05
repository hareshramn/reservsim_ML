cmake_minimum_required(VERSION 3.16)
project(reserv_ml_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RESERV_ENABLE_CUDA "Enable CUDA toolchain for GPU builds" OFF)
if(RESERV_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(sim_config
    src/config_parser.cpp
    src/pressure_assembly.cpp
    src/state_init.cpp
    src/transport.cpp
)
target_include_directories(sim_config PUBLIC include)

add_executable(sim_run
    src/main.cpp
)
target_include_directories(sim_run PRIVATE include)
target_link_libraries(sim_run PRIVATE sim_config)

enable_testing()

add_executable(config_parser_tests
    tests/config_parser_tests.cpp
)
target_include_directories(config_parser_tests PRIVATE include)
target_link_libraries(config_parser_tests PRIVATE sim_config)

add_test(NAME config_parser_tests COMMAND config_parser_tests)

add_executable(state_init_tests
    tests/state_init_tests.cpp
)
target_include_directories(state_init_tests PRIVATE include)
target_link_libraries(state_init_tests PRIVATE sim_config)

add_test(NAME state_init_tests COMMAND state_init_tests)

add_executable(pressure_assembly_tests
    tests/pressure_assembly_tests.cpp
)
target_include_directories(pressure_assembly_tests PRIVATE include)
target_link_libraries(pressure_assembly_tests PRIVATE sim_config)

add_test(NAME pressure_assembly_tests COMMAND pressure_assembly_tests)

add_executable(transport_tests
    tests/transport_tests.cpp
)
target_include_directories(transport_tests PRIVATE include)
target_link_libraries(transport_tests PRIVATE sim_config)

add_test(NAME transport_tests COMMAND transport_tests)

add_executable(sim_run_contract_tests
    tests/sim_run_contract_tests.cpp
)
target_include_directories(sim_run_contract_tests PRIVATE include)
target_link_libraries(sim_run_contract_tests PRIVATE sim_config)

add_test(NAME sim_run_contract_tests COMMAND sim_run_contract_tests)
