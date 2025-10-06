#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

// Forward declarations
struct Pattern;
struct Consequent;
struct Rule;

// Pattern represents a logical pattern with predicate and arguments
struct Pattern {
    std::string predicate;
    std::vector<std::string> args;
    
    Pattern() = default;
    Pattern(const std::string& pred, const std::vector<std::string>& arguments)
        : predicate(pred), args(arguments) {}
    
    bool operator==(const Pattern& other) const {
        return predicate == other.predicate && args == other.args;
    }
};

// Consequent represents the output of a rule
struct Consequent {
    std::string predicate;
    std::vector<std::string> args;
    
    Consequent() = default;
    Consequent(const std::string& pred, const std::vector<std::string>& arguments)
        : predicate(pred), args(arguments) {}
    
    bool operator==(const Consequent& other) const {
        return predicate == other.predicate && args == other.args;
    }
};

// Rule represents a logical rule with pattern, consequent, and bias
struct Rule {
    int id;
    Pattern pattern;
    Consequent consequent;
    float bias;
    
    Rule() = default;
    Rule(int ruleId, const Pattern& pat, const Consequent& cons, float ruleBias)
        : id(ruleId), pattern(pat), consequent(cons), bias(ruleBias) {}
};

// Hash function for Consequent to use in unordered_map
struct ConsequentHash {
    std::size_t operator()(const Consequent& c) const {
        std::size_t h1 = std::hash<std::string>{}(c.predicate);
        std::size_t h2 = 0;
        for (const auto& arg : c.args) {
            h2 ^= std::hash<std::string>{}(arg) + 0x9e3779b9 + (h2 << 6) + (h2 >> 2);
        }
        return h1 ^ (h2 << 1);
    }
};

// Utility functions for pattern matching and substitution
std::pair<float, std::unordered_map<std::string, std::string>> matchScore(
    const Pattern& query, const Pattern& pattern);

Consequent substitute(const Consequent& consequent, 
                     const std::unordered_map<std::string, std::string>& bindings);

// JSON parsing utilities
std::vector<Rule> loadRulesFromJSON(const std::string& filename);
void saveRulesToJSON(const std::vector<Rule>& rules, const std::string& filename);
