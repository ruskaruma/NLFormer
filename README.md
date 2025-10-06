# NLFormer

A C++ implementation of a Neural Logic Transformer that combines neural networks with symbolic logic reasoning. This project extends the MIT Neural Logic Machine approach with transformer attention mechanisms for logical inference.

## Overview

NLFormer is a neural-symbolic AI system that can perform logical reasoning using rules and facts. It uses attention mechanisms to determine which rules are most relevant for a given query, then applies those rules to derive conclusions.

## Features

- Neural logic integration combining neural networks with symbolic reasoning
- Transformer attention mechanisms for rule scoring
- Multi-layer inference for hierarchical reasoning
- Context-aware reasoning with multiple facts
- JSON-based rule configuration
- Python bindings for easy integration

## Architecture

The system has these main components:

- Engine: Core inference engine that processes rules and facts
- Attention Module: Softmax attention for rule scoring
- Pattern Matcher: Pattern matching with variable binding
- Type System: Data structures for rules, patterns, and consequents

## Installation

### Prerequisites

- C++17 compiler (GCC 7+ or Clang 5+)
- CMake 3.10+
- nlohmann/json library
- Google Test (for testing)

### Build Instructions

```bash
git clone <repository-url>
cd NLFormer
mkdir build
cd build
cmake ..
make
```

## Usage

### Basic Example

```cpp
#include "engine.hpp"
#include "types.hpp"

// Load rules from JSON
std::vector<Rule> rules = loadRulesFromJSON("examples/rules.json");
Engine engine(rules);

// Create a query
Pattern query("is", {"vehicle", "car"});

// Get results
auto results = engine.infer(query);
```

### Running the Demo

```bash
cd build
./demo
```

### Running Tests

```bash
cd build
./nlformer_tests
```

## Rule Configuration

Rules are defined in JSON format with the following structure:

```json
{
  "id": 1,
  "pattern": "(is ?x car)",
  "consequent": "(can ?x drive)",
  "bias": 0.0
}
```

- **id**: Unique identifier for the rule
- **pattern**: Input pattern with variables (prefixed with ?)
- **consequent**: Output pattern with variable substitution
- **bias**: Bias term for rule scoring

## API Reference

### Engine Class

```cpp
Engine(const std::vector<Rule>& rules)
std::vector<std::pair<Consequent, float>> infer(const Pattern& query)
std::vector<std::pair<Consequent, float>> inferContext(const std::vector<Pattern>& facts)
std::vector<std::pair<Consequent, float>> inferMultiLayer(const std::vector<Pattern>& initialFacts, size_t maxLayers)
```

### Utility Functions

```cpp
std::vector<Rule> loadRulesFromJSON(const std::string& filename)
void saveRulesToJSON(const std::vector<Rule>& rules, const std::string& filename)
std::vector<float> softmax(const std::vector<float>& scores)
```

## Project Structure

```
NLFormer/
├── include/
│   ├── attention.hpp      # Attention mechanism implementation
│   └── engine.hpp         # Core inference engine
├── src/
│   ├── attention.cpp      # Attention implementation
│   └── engine.cpp         # Engine implementation
├── examples/
│   └── rules.json         # Example rule configuration
├── CMakeLists.txt         # Build configuration
└── README.md              # This file
```

## Project Structure

```
NLFormer/
├── include/          # Header files
├── src/             # Source files
├── examples/        # Example applications
├── tests/           # Unit tests
├── benchmarks/      # Performance tests
└── python/          # Python bindings
```

## Testing

The project includes comprehensive unit tests using Google Test:

```bash
cd build
make nlformer_tests
./nlformer_tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Neural Logic Machines: Learning and Reasoning with First-Order Logic
- MIT Neural Logic Machine implementation
- Transformer attention mechanisms
