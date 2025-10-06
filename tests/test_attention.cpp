#include <gtest/gtest.h>
#include "../include/attention.hpp"
#include <vector>
#include <cmath>

class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(AttentionTest, BasicSoftmax) {
    std::vector<float> scores = {1.0f, 2.0f, 3.0f};
    auto result = softmax(scores);
    
    ASSERT_EQ(result.size(), 3);
    
    // Check that all values are positive
    for (float val : result) {
        EXPECT_GT(val, 0.0f);
    }
    
    // Check that values sum to 1.0
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
    
    // Check that the highest score gets the highest probability
    EXPECT_GT(result[2], result[1]);
    EXPECT_GT(result[1], result[0]);
}

TEST_F(AttentionTest, EmptyVector) {
    std::vector<float> scores;
    auto result = softmax(scores);
    
    EXPECT_TRUE(result.empty());
}

TEST_F(AttentionTest, SingleElement) {
    std::vector<float> scores = {5.0f};
    auto result = softmax(scores);
    
    ASSERT_EQ(result.size(), 1);
    EXPECT_NEAR(result[0], 1.0f, 0.001f);
}

TEST_F(AttentionTest, NegativeScores) {
    std::vector<float> scores = {-1.0f, -2.0f, -3.0f};
    auto result = softmax(scores);
    
    ASSERT_EQ(result.size(), 3);
    
    // Check that all values are positive
    for (float val : result) {
        EXPECT_GT(val, 0.0f);
    }
    
    // Check that values sum to 1.0
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
    
    // Check that the least negative score gets the highest probability
    EXPECT_GT(result[0], result[1]);
    EXPECT_GT(result[1], result[2]);
}

TEST_F(AttentionTest, LargeScores) {
    std::vector<float> scores = {100.0f, 101.0f, 102.0f};
    auto result = softmax(scores);
    
    ASSERT_EQ(result.size(), 3);
    
    // Check that all values are positive
    for (float val : result) {
        EXPECT_GT(val, 0.0f);
    }
    
    // Check that values sum to 1.0
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
    
    // Check that the highest score gets the highest probability
    EXPECT_GT(result[2], result[1]);
    EXPECT_GT(result[1], result[0]);
}

TEST_F(AttentionTest, IdenticalScores) {
    std::vector<float> scores = {1.0f, 1.0f, 1.0f};
    auto result = softmax(scores);
    
    ASSERT_EQ(result.size(), 3);
    
    // All values should be equal (1/3)
    for (float val : result) {
        EXPECT_NEAR(val, 1.0f / 3.0f, 0.001f);
    }
    
    // Check that values sum to 1.0
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
}

TEST_F(AttentionTest, ExtremeValues) {
    std::vector<float> scores = {0.0f, 1000.0f};
    auto result = softmax(scores);
    
    ASSERT_EQ(result.size(), 2);
    
    // The large value should dominate
    EXPECT_NEAR(result[0], 0.0f, 0.001f);
    EXPECT_NEAR(result[1], 1.0f, 0.001f);
}

TEST_F(AttentionTest, NumericalStability) {
    // Test with very large differences in scores
    std::vector<float> scores = {1.0f, 50.0f, 100.0f};
    auto result = softmax(scores);
    
    ASSERT_EQ(result.size(), 3);
    
    // Check that all values are positive
    for (float val : result) {
        EXPECT_GT(val, 0.0f);
    }
    
    // Check that values sum to 1.0
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
    
    // The highest score should dominate
    EXPECT_GT(result[2], result[1]);
    EXPECT_GT(result[1], result[0]);
}
