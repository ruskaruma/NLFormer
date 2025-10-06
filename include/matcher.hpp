#pragma once
#include "types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace NLFormer {

/**
 * Advanced pattern matching engine with variable binding
 * Supports fuzzy matching, confidence scoring, and variable substitution
 */
class PatternMatcher {
public:
    /**
     * Match a query pattern against a rule pattern
     * @param query The input pattern to match
     * @param pattern The rule pattern to match against
     * @return Pair of (confidence_score, variable_bindings)
     */
    static std::pair<float, std::unordered_map<std::string, std::string>> 
    matchScore(const Pattern& query, const Pattern& pattern);
    
    /**
     * Perform fuzzy matching with configurable threshold
     * @param query Input pattern
     * @param pattern Rule pattern
     * @param threshold Minimum confidence threshold
     * @return Match result with confidence and bindings
     */
    static std::pair<float, std::unordered_map<std::string, std::string>>
    fuzzyMatch(const Pattern& query, const Pattern& pattern, float threshold = 0.7f);
    
    /**
     * Check if two patterns are compatible (can potentially match)
     * @param query Input pattern
     * @param pattern Rule pattern
     * @return True if patterns are compatible
     */
    static bool isCompatible(const Pattern& query, const Pattern& pattern);
    
    /**
     * Extract variables from a pattern
     * @param pattern Input pattern
     * @return Vector of variable names
     */
    static std::vector<std::string> extractVariables(const Pattern& pattern);
    
    /**
     * Validate pattern syntax
     * @param pattern Pattern to validate
     * @return True if pattern is valid
     */
    static bool validatePattern(const Pattern& pattern);

private:
    /**
     * Internal matching algorithm with backtracking
     */
    static std::pair<float, std::unordered_map<std::string, std::string>>
    matchInternal(const Pattern& query, const Pattern& pattern, 
                  std::unordered_map<std::string, std::string> bindings = {});
    
    /**
     * Calculate confidence score based on match quality
     */
    static float calculateConfidence(const Pattern& query, const Pattern& pattern,
                                   const std::unordered_map<std::string, std::string>& bindings);
    
    /**
     * Check if a variable binding is consistent
     */
    static bool isConsistentBinding(const std::string& var, const std::string& value,
                                   const std::unordered_map<std::string, std::string>& bindings);
};

/**
 * Substitution engine for variable replacement
 */
class SubstitutionEngine {
public:
    /**
     * Substitute variables in a consequent with bindings
     * @param consequent The consequent to substitute
     * @param bindings Variable bindings
     * @return New consequent with variables substituted
     */
    static Consequent substitute(const Consequent& consequent,
                               const std::unordered_map<std::string, std::string>& bindings);
    
    /**
     * Substitute variables in a pattern
     * @param pattern The pattern to substitute
     * @param bindings Variable bindings
     * @return New pattern with variables substituted
     */
    static Pattern substitute(const Pattern& pattern,
                            const std::unordered_map<std::string, std::string>& bindings);
    
    /**
     * Check if a consequent contains any unbound variables
     * @param consequent The consequent to check
     * @param bindings Available bindings
     * @return True if all variables are bound
     */
    static bool isFullyBound(const Consequent& consequent,
                            const std::unordered_map<std::string, std::string>& bindings);

private:
    /**
     * Substitute variables in a string
     */
    static std::string substituteString(const std::string& str,
                                      const std::unordered_map<std::string, std::string>& bindings);
};

} // namespace NLFormer

// Global functions for backward compatibility
std::pair<float, std::unordered_map<std::string, std::string>> matchScore(
    const Pattern& query, const Pattern& pattern);

Consequent substitute(const Consequent& consequent, 
                     const std::unordered_map<std::string, std::string>& bindings);
