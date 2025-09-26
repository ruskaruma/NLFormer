#pragma once
#include"types.hpp"
#include <vector>
#include<unordered_map>

class Engine
{
    private:
    std::vector<Rule> rules;
    public:
    Engine(const std::vector<Rule>& rules); 
    std::vector<std::pair<Consequent, float>> infer(const Pattern& query);
    std::vector<std::pair<Consequent, float>> inferContext(const std::vector<Pattern>& facts);
    std::vector<std::pair<Consequent, float>> inferMultiLayer(const std::vector<Pattern>& initialFacts, size_t maxLayers);
};