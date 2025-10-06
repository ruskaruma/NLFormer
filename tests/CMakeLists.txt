# Test configuration
add_executable(nlformer_tests
    test_engine.cpp
    test_attention.cpp
    test_matcher.cpp
    test_types.cpp
)

target_link_libraries(nlformer_tests
    nlformer
    GTest::gtest
    GTest::gtest_main
)

# Add test to CTest
add_test(NAME nlformer_tests COMMAND nlformer_tests)

# Enable test discovery
include(GoogleTest)
gtest_discover_tests(nlformer_tests)
