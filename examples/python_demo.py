#!/usr/bin/env python3
"""
NLFormer Python Demo
====================

This demo showcases the Python bindings for NLFormer,
demonstrating neural logic reasoning with transformer attention.
"""

import nlformer_python as nlf
import json
import time
from typing import List, Tuple

def create_demo_rules() -> List[nlf.Rule]:
    """Create a set of demo rules for transportation reasoning."""
    rules = [
        nlf.Rule(1, nlf.Pattern("is", ["?x", "car"]), nlf.Consequent("can", ["?x", "drive"]), 0.0),
        nlf.Rule(2, nlf.Pattern("is", ["?x", "electricCar"]), nlf.Consequent("needs", ["?x", "fuel"]), -5.0),
        nlf.Rule(3, nlf.Pattern("is", ["?x", "damaged"]), nlf.Consequent("can", ["?x", "drive"]), -3.0),
        nlf.Rule(4, nlf.Pattern("can", ["?x", "drive"]), nlf.Consequent("needs", ["?x", "engine"]), 0.0),
        nlf.Rule(5, nlf.Pattern("needs", ["?x", "engine"]), nlf.Consequent("has", ["?x", "parts"]), 0.0),
        nlf.Rule(6, nlf.Pattern("is", ["?x", "truck"]), nlf.Consequent("can", ["?x", "carry"]), 0.0),
        nlf.Rule(7, nlf.Pattern("can", ["?x", "carry"]), nlf.Consequent("needs", ["?x", "cargo"]), 0.0),
    ]
    return rules

def print_results(results: List[Tuple[nlf.Consequent, float]], title: str):
    """Pretty print inference results."""
    print(f"\n{title}:")
    print("=" * (len(title) + 1))
    
    if not results:
        print("No results found.")
        return
    
    print(f"{'Consequent':<30} {'Weight':<15}")
    print("-" * 45)
    
    for consequent, weight in results:
        consequent_str = f"({consequent.predicate} {' '.join(consequent.args)})"
        print(f"{consequent_str:<30} {weight:<15.4f}")

def demo_basic_inference():
    """Demonstrate basic pattern inference."""
    print("NLFormer Python Demo - Basic Inference")
    print("=" * 42)
    
    # Create engine with demo rules
    rules = create_demo_rules()
    engine = nlf.Engine(rules)
    
    print(f"Created engine with {len(rules)} rules")
    
    # Test cases
    test_cases = [
        ("Car inference", nlf.Pattern("is", ["vehicle", "car"])),
        ("Electric car inference", nlf.Pattern("is", ["tesla", "electricCar"])),
        ("Damaged vehicle inference", nlf.Pattern("is", ["truck", "damaged"])),
        ("Truck inference", nlf.Pattern("is", ["bigTruck", "truck"])),
    ]
    
    for description, query in test_cases:
        print(f"\n{description}:")
        print(f"Query: ({query.predicate} {' '.join(query.args)})")
        
        start_time = time.time()
        results = engine.infer(query)
        end_time = time.time()
        
        print_results(results, "Results")
        print(f"Inference time: {(end_time - start_time) * 1000:.3f} ms")

def demo_context_inference():
    """Demonstrate context-aware inference."""
    print("\nNLFormer Python Demo - Context-Aware Inference")
    print("=" * 50)
    
    rules = create_demo_rules()
    engine = nlf.Engine(rules)
    
    # Context with multiple facts
    context = [
        nlf.Pattern("is", ["vehicle1", "car"]),
        nlf.Pattern("is", ["vehicle2", "electricCar"]),
        nlf.Pattern("is", ["vehicle3", "damaged"]),
        nlf.Pattern("is", ["vehicle4", "truck"]),
    ]
    
    print("Context facts:")
    for fact in context:
        print(f"  ({fact.predicate} {' '.join(fact.args)})")
    
    start_time = time.time()
    results = engine.infer_context(context)
    end_time = time.time()
    
    print_results(results, "Context Inference Results")
    print(f"Context inference time: {(end_time - start_time) * 1000:.3f} ms")

def demo_multi_layer_inference():
    """Demonstrate multi-layer reasoning."""
    print("\nNLFormer Python Demo - Multi-Layer Inference")
    print("=" * 45)
    
    rules = create_demo_rules()
    engine = nlf.Engine(rules)
    
    # Initial facts
    initial_facts = [
        nlf.Pattern("is", ["myCar", "car"]),
        nlf.Pattern("is", ["myTruck", "truck"]),
    ]
    
    print("Initial facts:")
    for fact in initial_facts:
        print(f"  ({fact.predicate} {' '.join(fact.args)})")
    
    print("\nPerforming multi-layer inference (max 3 layers)...")
    
    start_time = time.time()
    results = engine.infer_multi_layer(initial_facts, 3)
    end_time = time.time()
    
    print_results(results, "Multi-Layer Inference Results")
    print(f"Multi-layer inference time: {(end_time - start_time) * 1000:.3f} ms")

def demo_attention_mechanisms():
    """Demonstrate attention mechanisms."""
    print("\nNLFormer Python Demo - Attention Mechanisms")
    print("=" * 45)
    
    # Test softmax function
    scores = [1.0, 2.0, 3.0, 0.5, -1.0]
    attention_weights = nlf.softmax(scores)
    
    print("Input scores:", scores)
    print("Attention weights:", [f"{w:.4f}" for w in attention_weights])
    print("Sum of weights:", f"{sum(attention_weights):.6f}")
    
    # Test with extreme values
    extreme_scores = [100.0, 101.0, 102.0]
    extreme_weights = nlf.softmax(extreme_scores)
    
    print(f"\nExtreme scores: {extreme_scores}")
    print(f"Extreme weights: {[f'{w:.6f}' for w in extreme_weights]}")
    print(f"Sum of extreme weights: {sum(extreme_weights):.6f}")

def demo_performance():
    """Demonstrate performance characteristics."""
    print("\nNLFormer Python Demo - Performance Analysis")
    print("=" * 45)
    
    rules = create_demo_rules()
    engine = nlf.Engine(rules)
    
    # Performance test
    test_queries = [
        nlf.Pattern("is", ["car1", "car"]),
        nlf.Pattern("is", ["car2", "electricCar"]),
        nlf.Pattern("is", ["car3", "damaged"]),
        nlf.Pattern("can", ["car1", "drive"]),
        nlf.Pattern("needs", ["car1", "engine"]),
    ]
    
    iterations = 1000
    start_time = time.time()
    
    for i in range(iterations):
        for query in test_queries:
            results = engine.infer(query)
    
    end_time = time.time()
    
    total_queries = iterations * len(test_queries)
    total_time = end_time - start_time
    avg_time_per_query = total_time / total_queries
    
    print(f"Performance Results:")
    print(f"  Total queries: {total_queries}")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Average time per query: {avg_time_per_query * 1000:.3f} ms")
    print(f"  Queries per second: {1.0 / avg_time_per_query:.0f}")

def demo_json_integration():
    """Demonstrate JSON rule loading and saving."""
    print("\nNLFormer Python Demo - JSON Integration")
    print("=" * 40)
    
    # Create some rules
    rules = create_demo_rules()
    
    # Save rules to JSON
    nlf.save_rules_to_json(rules, "demo_rules.json")
    print("Saved rules to demo_rules.json")
    
    # Load rules from JSON
    loaded_rules = nlf.load_rules_from_json("demo_rules.json")
    print(f"Loaded {len(loaded_rules)} rules from JSON")
    
    # Create engine with loaded rules
    engine = nlf.Engine(loaded_rules)
    
    # Test inference
    query = nlf.Pattern("is", ["testVehicle", "car"])
    results = engine.infer(query)
    
    print_results(results, "JSON Loaded Rules Results")
    
    # Clean up
    import os
    os.remove("demo_rules.json")
    print("Cleaned up demo_rules.json")

def main():
    """Run the complete NLFormer Python demo."""
    print("NLFormer - Neural Logic Transformer Python Demo")
    print("=" * 50)
    print("A C++ implementation with Python bindings")
    print("for neural logic reasoning with transformer attention.\n")
    
    try:
        demo_basic_inference()
        demo_context_inference()
        demo_multi_layer_inference()
        demo_attention_mechanisms()
        demo_performance()
        demo_json_integration()
        
        print("\nPython demo completed successfully!")
        print("\nFor more information, visit: https://github.com/yourusername/NLFormer")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
