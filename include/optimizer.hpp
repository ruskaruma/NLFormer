#pragma once
#include "types.hpp"
#include "engine.hpp"
#include <vector>
#include <unordered_map>
#include <functional>
#include <random>

namespace NLFormer {

/**
 * Rule Learning and Optimization Engine
 * 
 * This class provides advanced features for learning new rules
 * and optimizing existing rule sets for better performance.
 */
class RuleOptimizer {
private:
    std::mt19937 rng;
    std::vector<Rule> rules;
    Engine engine;
    
public:
    RuleOptimizer(const std::vector<Rule>& initialRules) 
        : rng(std::random_device{}()), rules(initialRules), engine(initialRules) {}
    
    /**
     * Learn new rules from training data
     * @param trainingData Pairs of (input_pattern, expected_output)
     * @param maxRules Maximum number of new rules to learn
     * @return Vector of learned rules
     */
    std::vector<Rule> learnRules(
        const std::vector<std::pair<Pattern, Consequent>>& trainingData,
        size_t maxRules = 10
    );
    
    /**
     * Optimize rule weights using gradient descent
     * @param trainingData Training examples
     * @param learningRate Learning rate for optimization
     * @param epochs Number of training epochs
     */
    void optimizeWeights(
        const std::vector<std::pair<Pattern, Consequent>>& trainingData,
        float learningRate = 0.01f,
        size_t epochs = 100
    );
    
    /**
     * Remove redundant rules to improve performance
     * @param threshold Similarity threshold for redundancy detection
     * @return Optimized rule set
     */
    std::vector<Rule> removeRedundantRules(float threshold = 0.8f);
    
    /**
     * Merge similar rules to reduce rule set size
     * @param similarityThreshold Threshold for rule similarity
     * @return Merged rule set
     */
    std::vector<Rule> mergeSimilarRules(float similarityThreshold = 0.7f);
    
    /**
     * Generate rule statistics and performance metrics
     * @return Map of performance metrics
     */
    std::unordered_map<std::string, float> getRuleStatistics();
    
    /**
     * Validate rule set for logical consistency
     * @return True if rule set is consistent
     */
    bool validateRuleSet();
    
    /**
     * Get optimized engine with improved rules
     * @return Optimized engine
     */
    Engine getOptimizedEngine();

private:
    /**
     * Calculate rule similarity score
     */
    float calculateRuleSimilarity(const Rule& rule1, const Rule& rule2);
    
    /**
     * Generate candidate rules from patterns
     */
    std::vector<Rule> generateCandidateRules(const Pattern& pattern, const Consequent& consequent);
    
    /**
     * Calculate rule importance score
     */
    float calculateRuleImportance(const Rule& rule, const std::vector<std::pair<Pattern, Consequent>>& trainingData);
    
    /**
     * Optimize single rule weight
     */
    void optimizeRuleWeight(Rule& rule, const std::vector<std::pair<Pattern, Consequent>>& trainingData, float learningRate);
};

/**
 * Performance Profiler for NLFormer
 * 
 * Provides detailed performance analysis and optimization suggestions.
 */
class PerformanceProfiler {
private:
    Engine engine;
    std::vector<Rule> rules;
    
public:
    PerformanceProfiler(const Engine& eng, const std::vector<Rule>& ruleSet) 
        : engine(eng), rules(ruleSet) {}
    
    /**
     * Profile inference performance
     * @param testQueries Test queries for profiling
     * @return Performance metrics
     */
    std::unordered_map<std::string, float> profileInference(
        const std::vector<Pattern>& testQueries
    );
    
    /**
     * Analyze rule usage patterns
     * @param testQueries Test queries
     * @return Rule usage statistics
     */
    std::unordered_map<int, float> analyzeRuleUsage(
        const std::vector<Pattern>& testQueries
    );
    
    /**
     * Get optimization recommendations
     * @return Vector of optimization suggestions
     */
    std::vector<std::string> getOptimizationRecommendations();
    
    /**
     * Benchmark different rule set sizes
     * @param maxRules Maximum number of rules to test
     * @return Performance vs rule count data
     */
    std::vector<std::pair<int, float>> benchmarkRuleSetSizes(size_t maxRules = 100);
    
    /**
     * Memory usage analysis
     * @return Memory usage statistics
     */
    std::unordered_map<std::string, size_t> analyzeMemoryUsage();
};

/**
 * Advanced Pattern Matcher with Learning Capabilities
 * 
 * Extends the basic pattern matcher with machine learning features.
 */
class LearningPatternMatcher {
private:
    std::unordered_map<std::string, float> patternWeights;
    std::vector<std::pair<Pattern, Pattern>> learnedMappings;
    
public:
    LearningPatternMatcher() = default;
    
    /**
     * Learn pattern mappings from examples
     * @param examples Pairs of (input_pattern, target_pattern)
     */
    void learnMappings(const std::vector<std::pair<Pattern, Pattern>>& examples);
    
    /**
     * Enhanced pattern matching with learned weights
     * @param query Input pattern
     * @param pattern Target pattern
     * @return Enhanced match score and bindings
     */
    std::pair<float, std::unordered_map<std::string, std::string>> 
    enhancedMatch(const Pattern& query, const Pattern& pattern);
    
    /**
     * Get learned pattern weights
     * @return Map of pattern weights
     */
    const std::unordered_map<std::string, float>& getPatternWeights() const {
        return patternWeights;
    }
    
    /**
     * Save learned patterns to file
     * @param filename Output filename
     */
    void saveLearnedPatterns(const std::string& filename);
    
    /**
     * Load learned patterns from file
     * @param filename Input filename
     */
    void loadLearnedPatterns(const std::string& filename);

private:
    /**
     * Update pattern weights based on feedback
     */
    void updateWeights(const Pattern& pattern, float feedback);
    
    /**
     * Calculate pattern similarity
     */
    float calculatePatternSimilarity(const Pattern& p1, const Pattern& p2);
};

} // namespace NLFormer
