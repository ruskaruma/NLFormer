#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include "../include/engine.hpp"
#include "../include/types.hpp"

class BenchmarkSuite {
private:
    std::mt19937 rng;
    std::vector<Rule> rules;
    Engine engine;
    
public:
    BenchmarkSuite() : rng(std::random_device{}()) {
        // Create a comprehensive rule set for benchmarking
        rules = createBenchmarkRules();
        engine = Engine(rules);
    }
    
    std::vector<Rule> createBenchmarkRules() {
        std::vector<Rule> benchmarkRules;
        
        // Basic transportation rules
        benchmarkRules.emplace_back(1, Pattern("is", {"?x", "car"}), Consequent("can", {"?x", "drive"}), 0.0f);
        benchmarkRules.emplace_back(2, Pattern("is", {"?x", "electricCar"}), Consequent("needs", {"?x", "fuel"}), -5.0f);
        benchmarkRules.emplace_back(3, Pattern("is", {"?x", "damaged"}), Consequent("can", {"?x", "drive"}), -3.0f);
        benchmarkRules.emplace_back(4, Pattern("can", {"?x", "drive"}), Consequent("needs", {"?x", "engine"}), 0.0f);
        benchmarkRules.emplace_back(5, Pattern("needs", {"?x", "engine"}), Consequent("has", {"?x", "parts"}), 0.0f);
        
        // Add more complex rules for comprehensive testing
        for (int i = 6; i <= 50; ++i) {
            std::string predicate = "rule" + std::to_string(i);
            std::string consequentPred = "result" + std::to_string(i);
            benchmarkRules.emplace_back(
                i,
                Pattern(predicate, {"?x", "?y"}),
                Consequent(consequentPred, {"?x", "?y"}),
                static_cast<float>(i % 10 - 5)
            );
        }
        
        return benchmarkRules;
    }
    
    void runBasicInferenceBenchmark() {
        std::cout << "\nBasic Inference Benchmark\n";
        std::cout << "=========================\n";
        
        std::vector<Pattern> testQueries = {
            Pattern("is", {"vehicle", "car"}),
            Pattern("is", {"tesla", "electricCar"}),
            Pattern("is", {"truck", "damaged"}),
            Pattern("can", {"vehicle", "drive"}),
            Pattern("needs", {"vehicle", "engine"})
        };
        
        const int iterations = 10000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            for (const auto& query : testQueries) {
                auto results = engine.infer(query);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double totalQueries = iterations * testQueries.size();
        double avgTimePerQuery = static_cast<double>(duration.count()) / totalQueries;
        double queriesPerSecond = 1000000.0 / avgTimePerQuery;
        
        std::cout << "  Total queries: " << static_cast<int>(totalQueries) << "\n";
        std::cout << "  Total time: " << duration.count() << " μs\n";
        std::cout << "  Average time per query: " << std::fixed << std::setprecision(3) 
                  << avgTimePerQuery << " μs\n";
        std::cout << "  Queries per second: " << std::fixed << std::setprecision(0)
                  << queriesPerSecond << "\n";
    }
    
    void runContextInferenceBenchmark() {
        std::cout << "\nContext Inference Benchmark\n";
        std::cout << "===========================\n";
        
        std::vector<std::vector<Pattern>> testContexts = {
            {Pattern("is", {"car1", "car"}), Pattern("is", {"car2", "electricCar"})},
            {Pattern("is", {"vehicle", "damaged"}), Pattern("can", {"vehicle", "drive"})},
            {Pattern("needs", {"car", "engine"}), Pattern("has", {"car", "parts"})}
        };
        
        const int iterations = 5000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            for (const auto& context : testContexts) {
                auto results = engine.inferContext(context);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double totalContexts = iterations * testContexts.size();
        double avgTimePerContext = static_cast<double>(duration.count()) / totalContexts;
        double contextsPerSecond = 1000000.0 / avgTimePerContext;
        
        std::cout << "  Total contexts: " << static_cast<int>(totalContexts) << "\n";
        std::cout << "  Total time: " << duration.count() << " μs\n";
        std::cout << "  Average time per context: " << std::fixed << std::setprecision(3) 
                  << avgTimePerContext << " μs\n";
        std::cout << "  Contexts per second: " << std::fixed << std::setprecision(0)
                  << contextsPerSecond << "\n";
    }
    
    void runMultiLayerBenchmark() {
        std::cout << "\nMulti-Layer Inference Benchmark\n";
        std::cout << "================================\n";
        
        std::vector<Pattern> initialFacts = {
            Pattern("is", {"vehicle", "car"}),
            Pattern("is", {"vehicle2", "electricCar"})
        };
        
        const int iterations = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            auto results = engine.inferMultiLayer(initialFacts, 3);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avgTimePerInference = static_cast<double>(duration.count()) / iterations;
        double inferencesPerSecond = 1000000.0 / avgTimePerInference;
        
        std::cout << "  Total inferences: " << iterations << "\n";
        std::cout << "  Total time: " << duration.count() << " μs\n";
        std::cout << "  Average time per inference: " << std::fixed << std::setprecision(3) 
                  << avgTimePerInference << " μs\n";
        std::cout << "  Inferences per second: " << std::fixed << std::setprecision(0)
                  << inferencesPerSecond << "\n";
    }
    
    void runScalabilityBenchmark() {
        std::cout << "\nScalability Benchmark\n";
        std::cout << "====================\n";
        
        std::vector<int> ruleCounts = {10, 25, 50, 100, 200};
        
        for (int ruleCount : ruleCounts) {
            // Create engine with specified number of rules
            std::vector<Rule> testRules;
            for (int i = 1; i <= ruleCount; ++i) {
                testRules.emplace_back(
                    i,
                    Pattern("test" + std::to_string(i), {"?x", "?y"}),
                    Consequent("result" + std::to_string(i), {"?x", "?y"}),
                    static_cast<float>(i % 10)
                );
            }
            
            Engine testEngine(testRules);
            Pattern query("test1", {"arg1", "arg2"});
            
            const int iterations = 1000;
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iterations; ++i) {
                auto results = testEngine.infer(query);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            double avgTime = static_cast<double>(duration.count()) / iterations;
            
            std::cout << "  Rules: " << std::setw(3) << ruleCount 
                      << " | Avg time: " << std::fixed << std::setprecision(3) 
                      << avgTime << " μs\n";
        }
    }
    
    void runMemoryUsageBenchmark() {
        std::cout << "\nMemory Usage Analysis\n";
        std::cout << "====================\n";
        
        // Test with different rule set sizes
        std::vector<int> ruleCounts = {10, 50, 100, 500, 1000};
        
        for (int ruleCount : ruleCounts) {
            std::vector<Rule> testRules;
            for (int i = 1; i <= ruleCount; ++i) {
                testRules.emplace_back(
                    i,
                    Pattern("rule" + std::to_string(i), {"?x", "?y", "?z"}),
                    Consequent("result" + std::to_string(i), {"?x", "?y", "?z"}),
                    static_cast<float>(i % 10)
                );
            }
            
            Engine testEngine(testRules);
            
            // Estimate memory usage (rough calculation)
            size_t estimatedMemory = ruleCount * (sizeof(Rule) + 100); // Rough estimate
            
            std::cout << "  Rules: " << std::setw(4) << ruleCount 
                      << " | Estimated memory: " << std::setw(6) << estimatedMemory << " bytes\n";
        }
    }
    
    void runAllBenchmarks() {
        std::cout << "NLFormer Performance Benchmark Suite\n";
        std::cout << "====================================\n";
        std::cout << "Testing " << rules.size() << " rules\n";
        
        runBasicInferenceBenchmark();
        runContextInferenceBenchmark();
        runMultiLayerBenchmark();
        runScalabilityBenchmark();
        runMemoryUsageBenchmark();
        
        std::cout << "\nAll benchmarks completed!\n";
    }
};

int main() {
    try {
        BenchmarkSuite suite;
        suite.runAllBenchmarks();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
