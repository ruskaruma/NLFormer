#include "types.hpp"
#include "matcher.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Global function implementations
std::pair<float, std::unordered_map<std::string, std::string>> matchScore(
    const Pattern& query, const Pattern& pattern) {
    return NLFormer::PatternMatcher::matchScore(query, pattern);
}

Consequent substitute(const Consequent& consequent, 
                     const std::unordered_map<std::string, std::string>& bindings) {
    return NLFormer::SubstitutionEngine::substitute(consequent, bindings);
}

std::vector<Rule> loadRulesFromJSON(const std::string& filename) {
    std::vector<Rule> rules;
    
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        json j;
        file >> j;
        
        if (!j.is_array()) {
            throw std::runtime_error("JSON file must contain an array of rules");
        }
        
        for (const auto& ruleJson : j) {
            if (!ruleJson.contains("id") || !ruleJson.contains("pattern") || 
                !ruleJson.contains("consequent") || !ruleJson.contains("bias")) {
                throw std::runtime_error("Invalid rule format in JSON");
            }
            
            int id = ruleJson["id"];
            std::string patternStr = ruleJson["pattern"];
            std::string consequentStr = ruleJson["consequent"];
            float bias = ruleJson["bias"];
            
            // Parse pattern
            Pattern pattern = parsePattern(patternStr);
            
            // Parse consequent
            Consequent consequent = parseConsequent(consequentStr);
            
            rules.emplace_back(id, pattern, consequent, bias);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading rules from JSON: " << e.what() << std::endl;
        throw;
    }
    
    return rules;
}

void saveRulesToJSON(const std::vector<Rule>& rules, const std::string& filename) {
    try {
        json j = json::array();
        
        for (const auto& rule : rules) {
            json ruleJson;
            ruleJson["id"] = rule.id;
            ruleJson["pattern"] = patternToString(rule.pattern);
            ruleJson["consequent"] = consequentToString(rule.consequent);
            ruleJson["bias"] = rule.bias;
            
            j.push_back(ruleJson);
        }
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        file << j.dump(2) << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving rules to JSON: " << e.what() << std::endl;
        throw;
    }
}

// Helper functions for parsing
Pattern parsePattern(const std::string& patternStr) {
    // Remove parentheses and split
    std::string cleaned = patternStr;
    if (cleaned.front() == '(' && cleaned.back() == ')') {
        cleaned = cleaned.substr(1, cleaned.length() - 2);
    }
    
    std::istringstream iss(cleaned);
    std::string predicate;
    std::vector<std::string> args;
    
    iss >> predicate;
    
    std::string arg;
    while (iss >> arg) {
        args.push_back(arg);
    }
    
    return Pattern(predicate, args);
}

Consequent parseConsequent(const std::string& consequentStr) {
    // Remove parentheses and split
    std::string cleaned = consequentStr;
    if (cleaned.front() == '(' && cleaned.back() == ')') {
        cleaned = cleaned.substr(1, cleaned.length() - 2);
    }
    
    std::istringstream iss(cleaned);
    std::string predicate;
    std::vector<std::string> args;
    
    iss >> predicate;
    
    std::string arg;
    while (iss >> arg) {
        args.push_back(arg);
    }
    
    return Consequent(predicate, args);
}

std::string patternToString(const Pattern& pattern) {
    std::ostringstream oss;
    oss << "(" << pattern.predicate;
    for (const auto& arg : pattern.args) {
        oss << " " << arg;
    }
    oss << ")";
    return oss.str();
}

std::string consequentToString(const Consequent& consequent) {
    std::ostringstream oss;
    oss << "(" << consequent.predicate;
    for (const auto& arg : consequent.args) {
        oss << " " << arg;
    }
    oss << ")";
    return oss.str();
}
