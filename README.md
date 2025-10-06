# NLFormer

A high-performance C++ implementation of a Neural Logic Transformer that extends MIT's Neural Logic Machine approach with transformer attention mechanisms for advanced logical reasoning and inference.

## Overview

NLFormer combines the power of neural networks with symbolic logic reasoning, providing a robust framework for knowledge representation and inference. The system implements transformer-based attention mechanisms to enhance logical rule matching and consequence derivation, making it suitable for complex reasoning tasks in artificial intelligence applications.

## Features

- **Neural Logic Integration**: Combines neural network learning with symbolic logic reasoning
- **Transformer Attention**: Implements softmax-based attention mechanisms for rule scoring
- **Multi-layer Inference**: Supports hierarchical reasoning with configurable depth
- **Context-aware Reasoning**: Processes multiple facts simultaneously for comprehensive inference
- **High Performance**: Optimized C++ implementation for efficient computation
- **Flexible Rule System**: JSON-based rule configuration for easy customization

## Architecture

The NLFormer system consists of several key components:

- **Engine**: Core inference engine that processes rules and facts
- **Attention Module**: Implements softmax attention for rule scoring
- **Pattern Matching**: Advanced pattern matching with variable binding
- **Multi-layer Reasoning**: Hierarchical inference with configurable depth

## Installation

### Prerequisites

- C++17 compatible compiler (GCC 7+ or Clang 5+)
- CMake 3.10 or higher

### Build Instructions

```bash
# Clone the repository
git clone <repository-url>
cd NLFormer

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
make
```

## Usage

### Basic Inference

```cpp
#include "engine.hpp"
#include "types.hpp"

// Load rules from JSON configuration
std::vector<Rule> rules = loadRulesFromJSON("examples/rules.json");

// Create inference engine
Engine engine(rules);

// Define query pattern
Pattern query("is", {"car"});

// Perform inference
auto results = engine.infer(query);
```

### Multi-layer Reasoning

```cpp
// Define initial facts
std::vector<Pattern> facts = {
    Pattern("is", {"vehicle", "car"}),
    Pattern("is", {"vehicle", "damaged"})
};

// Perform multi-layer inference
auto results = engine.inferMultiLayer(facts, 3);
```

### Context-aware Inference

```cpp
// Process multiple facts in context
std::vector<Pattern> context = {
    Pattern("is", {"x", "car"}),
    Pattern("is", {"x", "electricCar"})
};

auto results = engine.inferContext(context);
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

#### Constructor
```cpp
Engine(const std::vector<Rule>& rules)
```

#### Methods

- `infer(const Pattern& query)`: Single pattern inference
- `inferContext(const std::vector<Pattern>& facts)`: Context-aware inference
- `inferMultiLayer(const std::vector<Pattern>& initialFacts, size_t maxLayers)`: Multi-layer reasoning

### Attention Module

- `softmax(const std::vector<float>& scores)`: Computes softmax attention weights

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

## Performance Considerations

- The system is optimized for high-performance inference with minimal memory overhead
- Multi-layer reasoning complexity scales with the number of rules and facts
- Attention computation uses numerical stability techniques for large score vectors

## Contributing

Contributions are welcome. Please ensure that:

- Code follows C++17 standards
- All new features include appropriate tests
- Documentation is updated for new functionality
- Code style is consistent with the existing codebase

## License

This project is licensed under the terms specified in the LICENSE file.

## References

- Neural Logic Machines: Learning and Reasoning with First-Order Logic
- MIT Neural Logic Machine implementation
- Transformer attention mechanisms for symbolic reasoning
