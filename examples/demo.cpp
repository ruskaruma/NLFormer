#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "../include/engine.hpp"
#include "../include/types.hpp"

void printResults(const std::vector<std::pair<Consequent, float>>& results, const std::string& title) {
    std::cout << "\n" << title << ":\n";
    std::cout << "=" << std::string(title.length() + 1, '=') << "\n";
    
    if (results.empty()) {
        std::cout << "No results found.\n";
        return;
    }
    
    std::cout << std::left << std::setw(30) << "Consequent" << std::setw(15) << "Weight" << "\n";
    std::cout << std::string(45, '-') << "\n";
    
    for (const auto& [consequent, weight] : results) {
        std::string consequentStr = "(" + consequent.predicate;
        for (const auto& arg : consequent.args) {
            consequentStr += " " + arg;
        }
        consequentStr += ")";
        
        std::cout << std::left << std::setw(30) << consequentStr 
                  << std::setw(15) << std::fixed << std::setprecision(4) << weight << "\n";
    }
}

void demonstrateBasicInference() {
    std::cout << "\nNLFormer Demo - Basic Inference\n";
    std::cout << "================================\n";
    
    // Load rules from JSON
    std::vector<Rule> rules;
    try {
        rules = loadRulesFromJSON("rules.json");
        std::cout << "Loaded " << rules.size() << " rules from rules.json\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading rules: " << e.what() << "\n";
        return;
    }
    
    // Create engine
    Engine engine(rules);
    
    // Test cases
    std::vector<std::pair<std::string, Pattern>> testCases = {
        {"Car inference", Pattern("is", {"vehicle", "car"})},
        {"Electric car inference", Pattern("is", {"tesla", "electricCar"})},
        {"Damaged vehicle inference", Pattern("is", {"truck", "damaged"})},
        {"Non-matching query", Pattern("is", {"plane", "aircraft"})}
    };
    
    for (const auto& [description, query] : testCases) {
        std::cout << "\n" << description << ":\n";
        std::cout << "Query: (" << query.predicate;
        for (const auto& arg : query.args) {
            std::cout << " " << arg;
        }
        std::cout << ")\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        auto results = engine.infer(query);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        printResults(results, "Results");
        std::cout << "Inference time: " << duration.count() << " μs\n";
    }
}

void demonstrateContextInference() {
    std::cout << "\nNLFormer Demo - Context-Aware Inference\n";
    std::cout << "======================================\n";
    
    std::vector<Rule> rules;
    try {
        rules = loadRulesFromJSON("rules.json");
    } catch (const std::exception& e) {
        std::cerr << "Error loading rules: " << e.what() << "\n";
        return;
    }
    
    Engine engine(rules);
    
    // Context with multiple facts
    std::vector<Pattern> context = {
        Pattern("is", {"vehicle1", "car"}),
        Pattern("is", {"vehicle2", "electricCar"}),
        Pattern("is", {"vehicle3", "damaged"})
    };
    
    std::cout << "Context facts:\n";
    for (const auto& fact : context) {
        std::cout << "  (" << fact.predicate;
        for (const auto& arg : fact.args) {
            std::cout << " " << arg;
        }
        std::cout << ")\n";
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = engine.inferContext(context);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printResults(results, "Context Inference Results");
    std::cout << "Context inference time: " << duration.count() << " μs\n";
}

void demonstrateMultiLayerInference() {
    std::cout << "\nNLFormer Demo - Multi-Layer Inference\n";
    std::cout << "=====================================\n";
    
    std::vector<Rule> rules;
    try {
        rules = loadRulesFromJSON("rules.json");
    } catch (const std::exception& e) {
        std::cerr << "Error loading rules: " << e.what() << "\n";
        return;
    }
    
    Engine engine(rules);
    
    // Initial facts
    std::vector<Pattern> initialFacts = {
        Pattern("is", {"myCar", "car"})
    };
    
    std::cout << "Initial facts:\n";
    for (const auto& fact : initialFacts) {
        std::cout << "  (" << fact.predicate;
        for (const auto& arg : fact.args) {
            std::cout << " " << arg;
        }
        std::cout << ")\n";
    }
    
    std::cout << "\nPerforming multi-layer inference (max 3 layers)...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = engine.inferMultiLayer(initialFacts, 3);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printResults(results, "Multi-Layer Inference Results");
    std::cout << "Multi-layer inference time: " << duration.count() << " μs\n";
}

void demonstratePerformance() {
    std::cout << "\nNLFormer Demo - Performance Analysis\n";
    std::cout << "====================================\n";
    
    std::vector<Rule> rules;
    try {
        rules = loadRulesFromJSON("rules.json");
    } catch (const std::exception& e) {
        std::cerr << "Error loading rules: " << e.what() << "\n";
        return;
    }
    
    Engine engine(rules);
    
    // Performance test with multiple queries
    std::vector<Pattern> queries = {
        Pattern("is", {"car1", "car"}),
        Pattern("is", {"car2", "electricCar"}),
        Pattern("is", {"car3", "damaged"}),
        Pattern("can", {"car1", "drive"}),
        Pattern("needs", {"car1", "engine"})
    };
    
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        for (const auto& query : queries) {
            auto results = engine.infer(query);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double avgTimePerQuery = static_cast<double>(totalDuration.count()) / (iterations * queries.size());
    
    std::cout << "Performance Results:\n";
    std::cout << "  Total queries: " << iterations * queries.size() << "\n";
    std::cout << "  Total time: " << totalDuration.count() << " ms\n";
    std::cout << "  Average time per query: " << std::fixed << std::setprecision(3) 
              << avgTimePerQuery << " ms\n";
    std::cout << "  Queries per second: " << std::fixed << std::setprecision(0)
              << 1000.0 / avgTimePerQuery << "\n";
}

int main() {
    std::cout << "NLFormer - Neural Logic Transformer Demo\n";
    std::cout << "========================================\n";
    std::cout << "A C++ implementation of neural logic reasoning\n";
    std::cout << "with transformer attention mechanisms.\n";
    
    try {
        demonstrateBasicInference();
        demonstrateContextInference();
        demonstrateMultiLayerInference();
        demonstratePerformance();
        
        std::cout << "\nDemo completed successfully!\n";
        std::cout << "\nFor more information, visit: https://github.com/yourusername/NLFormer\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nError during demo: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
