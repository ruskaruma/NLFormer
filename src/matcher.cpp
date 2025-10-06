#include "matcher.hpp"
#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <regex>

namespace NLFormer {

std::pair<float, std::unordered_map<std::string, std::string>> 
PatternMatcher::matchScore(const Pattern& query, const Pattern& pattern) {
    return matchInternal(query, pattern);
}

std::pair<float, std::unordered_map<std::string, std::string>>
PatternMatcher::fuzzyMatch(const Pattern& query, const Pattern& pattern, float threshold) {
    auto [score, bindings] = matchInternal(query, pattern);
    if (score >= threshold) {
        return {score, bindings};
    }
    return {0.0f, {}};
}

bool PatternMatcher::isCompatible(const Pattern& query, const Pattern& pattern) {
    if (query.predicate != pattern.predicate) {
        return false;
    }
    if (query.args.size() != pattern.args.size()) {
        return false;
    }
    return true;
}

std::vector<std::string> PatternMatcher::extractVariables(const Pattern& pattern) {
    std::vector<std::string> variables;
    for (const auto& arg : pattern.args) {
        if (arg.length() > 1 && arg[0] == '?') {
            variables.push_back(arg);
        }
    }
    return variables;
}

bool PatternMatcher::validatePattern(const Pattern& pattern) {
    if (pattern.predicate.empty()) {
        return false;
    }
    for (const auto& arg : pattern.args) {
        if (arg.empty()) {
            return false;
        }
    }
    return true;
}

std::pair<float, std::unordered_map<std::string, std::string>>
PatternMatcher::matchInternal(const Pattern& query, const Pattern& pattern, 
                              std::unordered_map<std::string, std::string> bindings) {
    if (!isCompatible(query, pattern)) {
        return {0.0f, {}};
    }
    
    float confidence = 1.0f;
    std::unordered_map<std::string, std::string> resultBindings = bindings;
    
    for (size_t i = 0; i < query.args.size(); ++i) {
        const std::string& queryArg = query.args[i];
        const std::string& patternArg = pattern.args[i];
        
        if (patternArg.length() > 1 && patternArg[0] == '?') {
            // Pattern argument is a variable
            std::string varName = patternArg;
            if (resultBindings.find(varName) != resultBindings.end()) {
                // Variable already bound, check consistency
                if (resultBindings[varName] != queryArg) {
                    return {0.0f, {}}; // Inconsistent binding
                }
            } else {
                // New variable binding
                resultBindings[varName] = queryArg;
            }
        } else {
            // Pattern argument is a literal, must match exactly
            if (queryArg != patternArg) {
                return {0.0f, {}}; // No match
            }
        }
    }
    
    confidence = calculateConfidence(query, pattern, resultBindings);
    return {confidence, resultBindings};
}

float PatternMatcher::calculateConfidence(const Pattern& query, const Pattern& pattern,
                                        const std::unordered_map<std::string, std::string>& bindings) {
    // Base confidence is 1.0 for exact matches
    float confidence = 1.0f;
    
    // Reduce confidence for fuzzy matches (could be extended)
    // For now, we only do exact matching
    return confidence;
}

bool PatternMatcher::isConsistentBinding(const std::string& var, const std::string& value,
                                        const std::unordered_map<std::string, std::string>& bindings) {
    auto it = bindings.find(var);
    if (it != bindings.end()) {
        return it->second == value;
    }
    return true; // Variable not bound yet
}

Consequent SubstitutionEngine::substitute(const Consequent& consequent,
                                        const std::unordered_map<std::string, std::string>& bindings) {
    Consequent result;
    result.predicate = consequent.predicate;
    
    for (const auto& arg : consequent.args) {
        if (arg.length() > 1 && arg[0] == '?') {
            // Variable substitution
            auto it = bindings.find(arg);
            if (it != bindings.end()) {
                result.args.push_back(it->second);
            } else {
                result.args.push_back(arg); // Keep original if not bound
            }
        } else {
            result.args.push_back(arg); // Literal, no substitution needed
        }
    }
    
    return result;
}

Pattern SubstitutionEngine::substitute(const Pattern& pattern,
                                      const std::unordered_map<std::string, std::string>& bindings) {
    Pattern result;
    result.predicate = pattern.predicate;
    
    for (const auto& arg : pattern.args) {
        if (arg.length() > 1 && arg[0] == '?') {
            // Variable substitution
            auto it = bindings.find(arg);
            if (it != bindings.end()) {
                result.args.push_back(it->second);
            } else {
                result.args.push_back(arg); // Keep original if not bound
            }
        } else {
            result.args.push_back(arg); // Literal, no substitution needed
        }
    }
    
    return result;
}

bool SubstitutionEngine::isFullyBound(const Consequent& consequent,
                                     const std::unordered_map<std::string, std::string>& bindings) {
    for (const auto& arg : consequent.args) {
        if (arg.length() > 1 && arg[0] == '?') {
            if (bindings.find(arg) == bindings.end()) {
                return false; // Unbound variable found
            }
        }
    }
    return true;
}

std::string SubstitutionEngine::substituteString(const std::string& str,
                                                const std::unordered_map<std::string, std::string>& bindings) {
    std::string result = str;
    for (const auto& [var, value] : bindings) {
        std::regex varRegex("\\" + var);
        result = std::regex_replace(result, varRegex, value);
    }
    return result;
}

} // namespace NLFormer

// Global function implementations for backward compatibility
std::pair<float, std::unordered_map<std::string, std::string>> matchScore(
    const Pattern& query, const Pattern& pattern) {
    return NLFormer::PatternMatcher::matchScore(query, pattern);
}

Consequent substitute(const Consequent& consequent, 
                     const std::unordered_map<std::string, std::string>& bindings) {
    return NLFormer::SubstitutionEngine::substitute(consequent, bindings);
}
