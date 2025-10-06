#include <gtest/gtest.h>
#include "../include/engine.hpp"
#include "../include/types.hpp"
#include "../include/matcher.hpp"
#include <vector>
#include <string>

class EngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test rules
        rules = {
            Rule(1, Pattern("is", {"?x", "car"}), Consequent("can", {"?x", "drive"}), 0.0f),
            Rule(2, Pattern("is", {"?x", "electricCar"}), Consequent("needs", {"?x", "fuel"}), -5.0f),
            Rule(3, Pattern("is", {"?x", "damaged"}), Consequent("can", {"?x", "drive"}), -3.0f),
            Rule(4, Pattern("can", {"?x", "drive"}), Consequent("needs", {"?x", "engine"}), 0.0f),
            Rule(5, Pattern("needs", {"?x", "engine"}), Consequent("has", {"?x", "parts"}), 0.0f)
        };
        engine = std::make_unique<Engine>(rules);
    }
    
    std::vector<Rule> rules;
    std::unique_ptr<Engine> engine;
};

TEST_F(EngineTest, BasicInference) {
    Pattern query("is", {"vehicle", "car"});
    auto results = engine->infer(query);
    
    ASSERT_FALSE(results.empty());
    
    // Check that we get the expected consequent
    bool foundCanDrive = false;
    for (const auto& [consequent, weight] : results) {
        if (consequent.predicate == "can" && consequent.args[0] == "vehicle" && consequent.args[1] == "drive") {
            foundCanDrive = true;
            EXPECT_GT(weight, 0.0f);
            break;
        }
    }
    EXPECT_TRUE(foundCanDrive);
}

TEST_F(EngineTest, ContextInference) {
    std::vector<Pattern> facts = {
        Pattern("is", {"vehicle", "car"}),
        Pattern("is", {"vehicle", "damaged"})
    };
    
    auto results = engine->inferContext(facts);
    
    ASSERT_FALSE(results.empty());
    
    // Should have combined results from both facts
    bool foundCanDrive = false;
    bool foundNeedsEngine = false;
    
    for (const auto& [consequent, weight] : results) {
        if (consequent.predicate == "can" && consequent.args[0] == "vehicle" && consequent.args[1] == "drive") {
            foundCanDrive = true;
        }
        if (consequent.predicate == "needs" && consequent.args[0] == "vehicle" && consequent.args[1] == "engine") {
            foundNeedsEngine = true;
        }
    }
    
    EXPECT_TRUE(foundCanDrive);
    EXPECT_TRUE(foundNeedsEngine);
}

TEST_F(EngineTest, MultiLayerInference) {
    std::vector<Pattern> initialFacts = {
        Pattern("is", {"vehicle", "car"})
    };
    
    auto results = engine->inferMultiLayer(initialFacts, 3);
    
    ASSERT_FALSE(results.empty());
    
    // Should have derived multiple levels of consequences
    bool foundCanDrive = false;
    bool foundNeedsEngine = false;
    bool foundHasParts = false;
    
    for (const auto& [consequent, weight] : results) {
        if (consequent.predicate == "can" && consequent.args[0] == "vehicle" && consequent.args[1] == "drive") {
            foundCanDrive = true;
        }
        if (consequent.predicate == "needs" && consequent.args[0] == "vehicle" && consequent.args[1] == "engine") {
            foundNeedsEngine = true;
        }
        if (consequent.predicate == "has" && consequent.args[0] == "vehicle" && consequent.args[1] == "parts") {
            foundHasParts = true;
        }
    }
    
    EXPECT_TRUE(foundCanDrive);
    EXPECT_TRUE(foundNeedsEngine);
    EXPECT_TRUE(foundHasParts);
}

TEST_F(EngineTest, NoMatchInference) {
    Pattern query("is", {"vehicle", "airplane"});
    auto results = engine->infer(query);
    
    // Should return empty results or low-confidence results
    bool hasHighConfidence = false;
    for (const auto& [consequent, weight] : results) {
        if (weight > 0.5f) {
            hasHighConfidence = true;
            break;
        }
    }
    EXPECT_FALSE(hasHighConfidence);
}

TEST_F(EngineTest, WeightedResults) {
    Pattern query("is", {"vehicle", "electricCar"});
    auto results = engine->infer(query);
    
    ASSERT_FALSE(results.empty());
    
    // Check that results are properly weighted
    float totalWeight = 0.0f;
    for (const auto& [consequent, weight] : results) {
        EXPECT_GE(weight, 0.0f);
        EXPECT_LE(weight, 1.0f);
        totalWeight += weight;
    }
    
    // Total weight should be approximately 1.0 (softmax normalization)
    EXPECT_NEAR(totalWeight, 1.0f, 0.01f);
}

TEST_F(EngineTest, EmptyRules) {
    std::vector<Rule> emptyRules;
    Engine emptyEngine(emptyRules);
    
    Pattern query("is", {"vehicle", "car"});
    auto results = emptyEngine.infer(query);
    
    EXPECT_TRUE(results.empty());
}

TEST_F(EngineTest, LargeRuleSet) {
    // Create a larger rule set
    std::vector<Rule> largeRules;
    for (int i = 0; i < 100; ++i) {
        largeRules.emplace_back(
            i, 
            Pattern("test", {"?x", "value" + std::to_string(i)}),
            Consequent("result", {"?x", "output" + std::to_string(i)}),
            static_cast<float>(i % 10)
        );
    }
    
    Engine largeEngine(largeRules);
    Pattern query("test", {"item", "value50"});
    auto results = largeEngine.infer(query);
    
    ASSERT_FALSE(results.empty());
    
    // Should find the matching rule
    bool foundMatch = false;
    for (const auto& [consequent, weight] : results) {
        if (consequent.predicate == "result" && consequent.args[0] == "item" && consequent.args[1] == "output50") {
            foundMatch = true;
            EXPECT_GT(weight, 0.0f);
            break;
        }
    }
    EXPECT_TRUE(foundMatch);
}
