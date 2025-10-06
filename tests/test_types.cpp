#include <gtest/gtest.h>
#include "../include/types.hpp"
#include <fstream>
#include <sstream>

class TypesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test JSON file
        testJsonContent = R"([
            {
                "id": 1,
                "pattern": "(is ?x car)",
                "consequent": "(can ?x drive)",
                "bias": 0.0
            },
            {
                "id": 2,
                "pattern": "(is ?x electricCar)",
                "consequent": "(needs ?x fuel)",
                "bias": -5.0
            }
        ])";
        
        std::ofstream file("test_rules.json");
        file << testJsonContent;
        file.close();
    }
    
    void TearDown() override {
        // Clean up test file
        std::remove("test_rules.json");
    }
    
    std::string testJsonContent;
};

TEST_F(TypesTest, PatternEquality) {
    Pattern p1("is", {"vehicle", "car"});
    Pattern p2("is", {"vehicle", "car"});
    Pattern p3("is", {"vehicle", "airplane"});
    
    EXPECT_EQ(p1, p2);
    EXPECT_NE(p1, p3);
}

TEST_F(TypesTest, ConsequentEquality) {
    Consequent c1("can", {"vehicle", "drive"});
    Consequent c2("can", {"vehicle", "drive"});
    Consequent c3("can", {"vehicle", "fly"});
    
    EXPECT_EQ(c1, c2);
    EXPECT_NE(c1, c3);
}

TEST_F(TypesTest, RuleConstruction) {
    Pattern pattern("is", {"?x", "car"});
    Consequent consequent("can", {"?x", "drive"});
    Rule rule(1, pattern, consequent, 0.5f);
    
    EXPECT_EQ(rule.id, 1);
    EXPECT_EQ(rule.pattern, pattern);
    EXPECT_EQ(rule.consequent, consequent);
    EXPECT_FLOAT_EQ(rule.bias, 0.5f);
}

TEST_F(TypesTest, ConsequentHash) {
    Consequent c1("can", {"vehicle", "drive"});
    Consequent c2("can", {"vehicle", "drive"});
    Consequent c3("can", {"vehicle", "fly"});
    
    ConsequentHash hasher;
    
    EXPECT_EQ(hasher(c1), hasher(c2));
    EXPECT_NE(hasher(c1), hasher(c3));
}

TEST_F(TypesTest, LoadRulesFromJSON) {
    auto rules = loadRulesFromJSON("test_rules.json");
    
    ASSERT_EQ(rules.size(), 2);
    
    // Check first rule
    EXPECT_EQ(rules[0].id, 1);
    EXPECT_EQ(rules[0].pattern.predicate, "is");
    EXPECT_EQ(rules[0].pattern.args[0], "?x");
    EXPECT_EQ(rules[0].pattern.args[1], "car");
    EXPECT_EQ(rules[0].consequent.predicate, "can");
    EXPECT_EQ(rules[0].consequent.args[0], "?x");
    EXPECT_EQ(rules[0].consequent.args[1], "drive");
    EXPECT_FLOAT_EQ(rules[0].bias, 0.0f);
    
    // Check second rule
    EXPECT_EQ(rules[1].id, 2);
    EXPECT_EQ(rules[1].pattern.predicate, "is");
    EXPECT_EQ(rules[1].pattern.args[0], "?x");
    EXPECT_EQ(rules[1].pattern.args[1], "electricCar");
    EXPECT_EQ(rules[1].consequent.predicate, "needs");
    EXPECT_EQ(rules[1].consequent.args[0], "?x");
    EXPECT_EQ(rules[1].consequent.args[1], "fuel");
    EXPECT_FLOAT_EQ(rules[1].bias, -5.0f);
}

TEST_F(TypesTest, SaveRulesToJSON) {
    std::vector<Rule> rules = {
        Rule(1, Pattern("is", {"?x", "car"}), Consequent("can", {"?x", "drive"}), 0.0f),
        Rule(2, Pattern("is", {"?x", "electricCar"}), Consequent("needs", {"?x", "fuel"}), -5.0f)
    };
    
    saveRulesToJSON(rules, "output_rules.json");
    
    // Verify the file was created and contains expected content
    std::ifstream file("output_rules.json");
    ASSERT_TRUE(file.is_open());
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    EXPECT_TRUE(content.find("\"id\": 1") != std::string::npos);
    EXPECT_TRUE(content.find("\"id\": 2") != std::string::npos);
    EXPECT_TRUE(content.find("(is ?x car)") != std::string::npos);
    EXPECT_TRUE(content.find("(can ?x drive)") != std::string::npos);
    
    file.close();
    std::remove("output_rules.json");
}

TEST_F(TypesTest, InvalidJSONFile) {
    EXPECT_THROW(loadRulesFromJSON("nonexistent.json"), std::runtime_error);
}

TEST_F(TypesTest, MalformedJSON) {
    std::ofstream file("malformed.json");
    file << "invalid json content";
    file.close();
    
    EXPECT_THROW(loadRulesFromJSON("malformed.json"), std::runtime_error);
    
    std::remove("malformed.json");
}

TEST_F(TypesTest, EmptyJSONArray) {
    std::ofstream file("empty.json");
    file << "[]";
    file.close();
    
    auto rules = loadRulesFromJSON("empty.json");
    EXPECT_TRUE(rules.empty());
    
    std::remove("empty.json");
}

TEST_F(TypesTest, JSONWithMissingFields) {
    std::ofstream file("incomplete.json");
    file << R"([{"id": 1, "pattern": "(is ?x car)"}])";
    file.close();
    
    EXPECT_THROW(loadRulesFromJSON("incomplete.json"), std::runtime_error);
    
    std::remove("incomplete.json");
}
