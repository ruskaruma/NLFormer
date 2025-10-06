#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/engine.hpp"
#include "../include/types.hpp"
#include "../include/attention.hpp"
#include <vector>
#include <string>

namespace py = pybind11;

// Python-friendly wrapper classes
class PyPattern {
public:
    std::string predicate;
    std::vector<std::string> args;
    
    PyPattern(const std::string& pred, const std::vector<std::string>& arguments)
        : predicate(pred), args(arguments) {}
    
    Pattern toPattern() const {
        return Pattern(predicate, args);
    }
    
    static PyPattern fromPattern(const Pattern& pattern) {
        return PyPattern(pattern.predicate, pattern.args);
    }
};

class PyConsequent {
public:
    std::string predicate;
    std::vector<std::string> args;
    
    PyConsequent(const std::string& pred, const std::vector<std::string>& arguments)
        : predicate(pred), args(arguments) {}
    
    Consequent toConsequent() const {
        return Consequent(predicate, args);
    }
    
    static PyConsequent fromConsequent(const Consequent& consequent) {
        return PyConsequent(consequent.predicate, consequent.args);
    }
};

class PyRule {
public:
    int id;
    PyPattern pattern;
    PyConsequent consequent;
    float bias;
    
    PyRule(int ruleId, const PyPattern& pat, const PyConsequent& cons, float ruleBias)
        : id(ruleId), pattern(pat), consequent(cons), bias(ruleBias) {}
    
    Rule toRule() const {
        return Rule(id, pattern.toPattern(), consequent.toConsequent(), bias);
    }
    
    static PyRule fromRule(const Rule& rule) {
        return PyRule(rule.id, PyPattern::fromPattern(rule.pattern), 
                     PyConsequent::fromConsequent(rule.consequent), rule.bias);
    }
};

class PyEngine {
private:
    Engine engine;
    
public:
    PyEngine(const std::vector<PyRule>& rules) {
        std::vector<Rule> cppRules;
        for (const auto& rule : rules) {
            cppRules.push_back(rule.toRule());
        }
        engine = Engine(cppRules);
    }
    
    std::vector<std::pair<PyConsequent, float>> infer(const PyPattern& query) {
        auto results = engine.infer(query.toPattern());
        std::vector<std::pair<PyConsequent, float>> pyResults;
        
        for (const auto& [consequent, weight] : results) {
            pyResults.emplace_back(PyConsequent::fromConsequent(consequent), weight);
        }
        
        return pyResults;
    }
    
    std::vector<std::pair<PyConsequent, float>> inferContext(const std::vector<PyPattern>& facts) {
        std::vector<Pattern> cppFacts;
        for (const auto& fact : facts) {
            cppFacts.push_back(fact.toPattern());
        }
        
        auto results = engine.inferContext(cppFacts);
        std::vector<std::pair<PyConsequent, float>> pyResults;
        
        for (const auto& [consequent, weight] : results) {
            pyResults.emplace_back(PyConsequent::fromConsequent(consequent), weight);
        }
        
        return pyResults;
    }
    
    std::vector<std::pair<PyConsequent, float>> inferMultiLayer(const std::vector<PyPattern>& initialFacts, size_t maxLayers) {
        std::vector<Pattern> cppFacts;
        for (const auto& fact : initialFacts) {
            cppFacts.push_back(fact.toPattern());
        }
        
        auto results = engine.inferMultiLayer(cppFacts, maxLayers);
        std::vector<std::pair<PyConsequent, float>> pyResults;
        
        for (const auto& [consequent, weight] : results) {
            pyResults.emplace_back(PyConsequent::fromConsequent(consequent), weight);
        }
        
        return pyResults;
    }
};

// Utility functions
std::vector<PyRule> loadRulesFromJSON(const std::string& filename) {
    auto cppRules = ::loadRulesFromJSON(filename);
    std::vector<PyRule> pyRules;
    
    for (const auto& rule : cppRules) {
        pyRules.push_back(PyRule::fromRule(rule));
    }
    
    return pyRules;
}

void saveRulesToJSON(const std::vector<PyRule>& rules, const std::string& filename) {
    std::vector<Rule> cppRules;
    for (const auto& rule : rules) {
        cppRules.push_back(rule.toRule());
    }
    
    ::saveRulesToJSON(cppRules, filename);
}

std::vector<float> softmax(const std::vector<float>& scores) {
    return ::softmax(scores);
}

PYBIND11_MODULE(nlformer_python, m) {
    m.doc() = "NLFormer: Neural Logic Transformer with Python bindings";
    
    // Pattern class
    py::class_<PyPattern>(m, "Pattern")
        .def(py::init<const std::string&, const std::vector<std::string>&>())
        .def_readwrite("predicate", &PyPattern::predicate)
        .def_readwrite("args", &PyPattern::args)
        .def("__repr__", [](const PyPattern& p) {
            std::string result = "Pattern(" + p.predicate + ", [";
            for (size_t i = 0; i < p.args.size(); ++i) {
                if (i > 0) result += ", ";
                result += "\"" + p.args[i] + "\"";
            }
            result += "])";
            return result;
        });
    
    // Consequent class
    py::class_<PyConsequent>(m, "Consequent")
        .def(py::init<const std::string&, const std::vector<std::string>&>())
        .def_readwrite("predicate", &PyConsequent::predicate)
        .def_readwrite("args", &PyConsequent::args)
        .def("__repr__", [](const PyConsequent& c) {
            std::string result = "Consequent(" + c.predicate + ", [";
            for (size_t i = 0; i < c.args.size(); ++i) {
                if (i > 0) result += ", ";
                result += "\"" + c.args[i] + "\"";
            }
            result += "])";
            return result;
        });
    
    // Rule class
    py::class_<PyRule>(m, "Rule")
        .def(py::init<int, const PyPattern&, const PyConsequent&, float>())
        .def_readwrite("id", &PyRule::id)
        .def_readwrite("pattern", &PyRule::pattern)
        .def_readwrite("consequent", &PyRule::consequent)
        .def_readwrite("bias", &PyRule::bias)
        .def("__repr__", [](const PyRule& r) {
            return "Rule(id=" + std::to_string(r.id) + ", bias=" + std::to_string(r.bias) + ")";
        });
    
    // Engine class
    py::class_<PyEngine>(m, "Engine")
        .def(py::init<const std::vector<PyRule>&>())
        .def("infer", &PyEngine::infer, "Perform single pattern inference")
        .def("infer_context", &PyEngine::inferContext, "Perform context-aware inference")
        .def("infer_multi_layer", &PyEngine::inferMultiLayer, "Perform multi-layer inference");
    
    // Utility functions
    m.def("load_rules_from_json", &loadRulesFromJSON, "Load rules from JSON file");
    m.def("save_rules_to_json", &saveRulesToJSON, "Save rules to JSON file");
    m.def("softmax", &softmax, "Compute softmax attention weights");
    
    // Version info
    m.attr("__version__") = "1.0.0";
}
