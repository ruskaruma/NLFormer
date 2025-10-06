#include <gtest/gtest.h>
#include "../include/matcher.hpp"
#include "../include/types.hpp"

class MatcherTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(MatcherTest, BasicPatternMatching) {
    Pattern query("is", {"vehicle", "car"});
    Pattern pattern("is", {"?x", "car"});
    
    auto [score, bindings] = NLFormer::PatternMatcher::matchScore(query, pattern);
    
    EXPECT_GT(score, 0.0f);
    EXPECT_EQ(bindings["?x"], "vehicle");
}

TEST_F(MatcherTest, NoMatch) {
    Pattern query("is", {"vehicle", "airplane"});
    Pattern pattern("is", {"?x", "car"});
    
    auto [score, bindings] = NLFormer::PatternMatcher::matchScore(query, pattern);
    
    EXPECT_EQ(score, 0.0f);
    EXPECT_TRUE(bindings.empty());
}

TEST_F(MatcherTest, MultipleVariables) {
    Pattern query("relation", {"A", "B", "C"});
    Pattern pattern("relation", {"?x", "?y", "?z"});
    
    auto [score, bindings] = NLFormer::PatternMatcher::matchScore(query, pattern);
    
    EXPECT_GT(score, 0.0f);
    EXPECT_EQ(bindings["?x"], "A");
    EXPECT_EQ(bindings["?y"], "B");
    EXPECT_EQ(bindings["?z"], "C");
}

TEST_F(MatcherTest, MixedLiteralsAndVariables) {
    Pattern query("parent", {"John", "Mary"});
    Pattern pattern("parent", {"?x", "?y"});
    
    auto [score, bindings] = NLFormer::PatternMatcher::matchScore(query, pattern);
    
    EXPECT_GT(score, 0.0f);
    EXPECT_EQ(bindings["?x"], "John");
    EXPECT_EQ(bindings["?y"], "Mary");
}

TEST_F(MatcherTest, InconsistentBinding) {
    Pattern query("relation", {"A", "A"});
    Pattern pattern("relation", {"?x", "?y"});
    
    auto [score, bindings] = NLFormer::PatternMatcher::matchScore(query, pattern);
    
    EXPECT_GT(score, 0.0f);
    EXPECT_EQ(bindings["?x"], "A");
    EXPECT_EQ(bindings["?y"], "A");
}

TEST_F(MatcherTest, FuzzyMatching) {
    Pattern query("is", {"vehicle", "car"});
    Pattern pattern("is", {"?x", "car"});
    
    auto [score, bindings] = NLFormer::PatternMatcher::fuzzyMatch(query, pattern, 0.5f);
    
    EXPECT_GT(score, 0.5f);
    EXPECT_EQ(bindings["?x"], "vehicle");
}

TEST_F(MatcherTest, FuzzyMatchingBelowThreshold) {
    Pattern query("is", {"vehicle", "airplane"});
    Pattern pattern("is", {"?x", "car"});
    
    auto [score, bindings] = NLFormer::PatternMatcher::fuzzyMatch(query, pattern, 0.9f);
    
    EXPECT_EQ(score, 0.0f);
    EXPECT_TRUE(bindings.empty());
}

TEST_F(MatcherTest, PatternCompatibility) {
    Pattern query("is", {"vehicle", "car"});
    Pattern compatible("is", {"?x", "?y"});
    Pattern incompatible("has", {"?x", "?y"});
    
    EXPECT_TRUE(NLFormer::PatternMatcher::isCompatible(query, compatible));
    EXPECT_FALSE(NLFormer::PatternMatcher::isCompatible(query, incompatible));
}

TEST_F(MatcherTest, ExtractVariables) {
    Pattern pattern("relation", {"?x", "literal", "?y"});
    auto variables = NLFormer::PatternMatcher::extractVariables(pattern);
    
    ASSERT_EQ(variables.size(), 2);
    EXPECT_EQ(variables[0], "?x");
    EXPECT_EQ(variables[1], "?y");
}

TEST_F(MatcherTest, ValidatePattern) {
    Pattern valid("predicate", {"arg1", "arg2"});
    Pattern invalid("", {"arg1"});
    Pattern invalid2("predicate", {""});
    
    EXPECT_TRUE(NLFormer::PatternMatcher::validatePattern(valid));
    EXPECT_FALSE(NLFormer::PatternMatcher::validatePattern(invalid));
    EXPECT_FALSE(NLFormer::PatternMatcher::validatePattern(invalid2));
}

TEST_F(MatcherTest, SubstitutionEngine) {
    Consequent consequent("can", {"?x", "drive"});
    std::unordered_map<std::string, std::string> bindings = {{"?x", "vehicle"}};
    
    auto result = NLFormer::SubstitutionEngine::substitute(consequent, bindings);
    
    EXPECT_EQ(result.predicate, "can");
    EXPECT_EQ(result.args[0], "vehicle");
    EXPECT_EQ(result.args[1], "drive");
}

TEST_F(MatcherTest, PatternSubstitution) {
    Pattern pattern("is", {"?x", "car"});
    std::unordered_map<std::string, std::string> bindings = {{"?x", "vehicle"}};
    
    auto result = NLFormer::SubstitutionEngine::substitute(pattern, bindings);
    
    EXPECT_EQ(result.predicate, "is");
    EXPECT_EQ(result.args[0], "vehicle");
    EXPECT_EQ(result.args[1], "car");
}

TEST_F(MatcherTest, FullyBoundCheck) {
    Consequent fullyBound("can", {"vehicle", "drive"});
    Consequent partiallyBound("can", {"?x", "drive"});
    std::unordered_map<std::string, std::string> bindings = {{"?x", "vehicle"}};
    
    EXPECT_TRUE(NLFormer::SubstitutionEngine::isFullyBound(fullyBound, bindings));
    EXPECT_TRUE(NLFormer::SubstitutionEngine::isFullyBound(partiallyBound, bindings));
    
    std::unordered_map<std::string, std::string> emptyBindings;
    EXPECT_FALSE(NLFormer::SubstitutionEngine::isFullyBound(partiallyBound, emptyBindings));
}
