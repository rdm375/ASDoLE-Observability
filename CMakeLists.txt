# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only

cmake_minimum_required(VERSION 3.20)
project(asdole LANGUAGES CXX)

include(CTest)

add_library(asdole INTERFACE)
target_include_directories(asdole INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(asdole INTERFACE cxx_std_20)

option(ASDOLE_BUILD_EXAMPLE "Build the scalar decay example" ON)
if(ASDOLE_BUILD_EXAMPLE)
    add_executable(asdole_decay examples/decay.cpp)
    target_link_libraries(asdole_decay PRIVATE asdole)

    add_executable(asdole_decay_control_convergence examples/decay_control_convergence.cpp)
    target_link_libraries(asdole_decay_control_convergence PRIVATE asdole)

    add_executable(asdole_nonlinear_adjoint_check examples/nonlinear_adjoint_check.cpp)
    target_link_libraries(asdole_nonlinear_adjoint_check PRIVATE asdole)

    add_executable(asdole_nonlinear_convergence examples/nonlinear_convergence.cpp)
    target_link_libraries(asdole_nonlinear_convergence PRIVATE asdole)

    add_executable(asdole_scalar_nonlinear_convergence examples/scalar_nonlinear_convergence.cpp)
    target_link_libraries(asdole_scalar_nonlinear_convergence PRIVATE asdole)

    add_executable(asdole_nonlinear_parameter_gradient_check examples/nonlinear_parameter_gradient_check.cpp)
    target_link_libraries(asdole_nonlinear_parameter_gradient_check PRIVATE asdole)

    add_executable(asdole_adaptive_decay examples/adaptive_decay.cpp)
    target_link_libraries(asdole_adaptive_decay PRIVATE asdole)

    add_executable(asdole_adaptive_norm_check examples/adaptive_norm_check.cpp)
    target_link_libraries(asdole_adaptive_norm_check PRIVATE asdole)

    add_executable(asdole_adaptive_transient examples/adaptive_transient.cpp)
    target_link_libraries(asdole_adaptive_transient PRIVATE asdole)

    add_executable(asdole_stability_sweep examples/stability_sweep.cpp)
    target_link_libraries(asdole_stability_sweep PRIVATE asdole)

    add_executable(asdole_nonnormal_stability examples/nonnormal_stability.cpp)
    target_link_libraries(asdole_nonnormal_stability PRIVATE asdole)

    add_executable(asdole_observation_analysis examples/observation_analysis.cpp)
    target_link_libraries(asdole_observation_analysis PRIVATE asdole)

    add_executable(asdole_parameter_identifiability examples/parameter_identifiability.cpp)
    target_link_libraries(asdole_parameter_identifiability PRIVATE asdole)

    add_executable(asdole_inversion_robustness examples/inversion_robustness.cpp)
    target_link_libraries(asdole_inversion_robustness PRIVATE asdole)

    add_executable(asdole_cross_coupled_error examples/cross_coupled_error.cpp)
    target_link_libraries(asdole_cross_coupled_error PRIVATE asdole)

    if(BUILD_TESTING)
        foreach(theta IN ITEMS 0.5 0.6 0.7 1.0)
            string(REPLACE "." "_" theta_test_suffix "${theta}")
            add_test(
                NAME nonlinear_stage_adjoint_theta_${theta_test_suffix}
                COMMAND asdole_nonlinear_adjoint_check "${theta}"
            )
        endforeach()
        foreach(theta IN ITEMS 0.5 0.6 0.7 1.0)
            string(REPLACE "." "_" theta_test_suffix "${theta}")
            add_test(
                NAME nonlinear_parameter_gradient_theta_${theta_test_suffix}
                COMMAND asdole_nonlinear_parameter_gradient_check "${theta}"
            )
        endforeach()
        add_test(NAME nonlinear_vector_second_order_convergence COMMAND asdole_nonlinear_convergence)
        add_test(NAME nonlinear_scalar_second_order_convergence COMMAND asdole_scalar_nonlinear_convergence)
        add_test(NAME adaptive_decay_step_control_default_inf_norm COMMAND asdole_adaptive_decay)
        add_test(NAME adaptive_decay_step_control_rms_norm COMMAND asdole_adaptive_decay --norm rms)
        add_test(NAME adaptive_error_norm_semantics COMMAND asdole_adaptive_norm_check)
        add_test(NAME adaptive_transient_step_response COMMAND asdole_adaptive_transient)
        add_test(NAME linear_state_tlm_adjoint_stability COMMAND asdole_stability_sweep)
        add_test(NAME nonnormal_tlm_adjoint_transient_growth COMMAND asdole_nonnormal_stability)
        add_test(NAME inverse_problem_noisy_parameter_recovery COMMAND asdole_inversion_robustness)
    endif()
endif()
