#include"engine.hpp"
#include"attention.hpp"
#include"matcher.hpp"
#include"types.hpp"
#include<algorithm>
#include<unordered_set>
Engine::Engine(const std::vector<Rule>& rules):rules(rules){}
std::vector<std::pair<Consequent, float>> Engine::infer(const Pattern& query)
{
    std::vector<float> scores;
    std::vector<std::unordered_map<std::string, std::string>> bindingsList;    
    for(const auto& rule : rules)
    {
        auto [score, bindings] = matchScore(query, rule.pattern);
        scores.push_back(score + rule.bias);
        bindingsList.push_back(bindings);
    }
    std::vector<float> weights = softmax(scores);
    std::vector<std::pair<Consequent, float>> result;
    for(size_t i = 0; i < rules.size(); ++i)
    {
        Consequent subConsequent = substitute(rules[i].consequent, bindingsList[i]);
        result.emplace_back(subConsequent, weights[i]);
    }
    return result;
}
std::vector<std::pair<Consequent, float>> Engine::inferContext(const std::vector<Pattern>& facts) {
    std::vector<std::pair<Consequent, float>> allConsequents;
    
    for(const auto& fact : facts)
    {
        auto res=infer(fact);
        allConsequents.insert(allConsequents.end(), res.begin(), res.end());
    }
    std::unordered_map<Consequent, float, ConsequentHash> map;
    for(const auto& [consequent, weight] : allConsequents)
    {
        map[consequent]+=weight;
    }
    std::vector<std::pair<Consequent, float>> result;
    for(const auto& [consequent, weight]:map)
    {
        result.emplace_back(consequent, weight);
    }
    return result;
}
std::vector<std::pair<Consequent, float>> Engine::inferMultiLayer(
    const std::vector<Pattern>& initialFacts, 
    size_t maxLayers)
    {
    std::vector<Pattern> knownFacts=initialFacts;
    std::unordered_map<Consequent, float, ConsequentHash> allConsequents;
    size_t layer=0;
    while(layer < maxLayers)
    {
        std::vector<Pattern>newFacts;
        for(const auto& fact : knownFacts)
        {
            for(const auto& rule : rules)
            {
                auto [score, bindings] = matchScore(fact, rule.pattern);
                if(score>0.0f){
                    Consequent consSub=substitute(rule.consequent, bindings);
                    allConsequents[consSub]+=score+rule.bias;
                    Pattern newPattern(consSub.predicate, consSub.args);    
                    bool alreadyKnown = std::find(knownFacts.begin(), knownFacts.end(), newPattern) != knownFacts.end();
                    bool alreadyNew = std::find(newFacts.begin(), newFacts.end(), newPattern) != newFacts.end();
                    if(!alreadyKnown && !alreadyNew)
                    {
                        newFacts.push_back(newPattern);
                    }
                }
            }
        }
        if(newFacts.empty()) break;        
        knownFacts.insert(knownFacts.end(), newFacts.begin(), newFacts.end());
        layer++;
    }
    std::vector<std::pair<Consequent, float>> result;
    for (const auto& [consequent, weight] : allConsequents)
    {
        result.emplace_back(consequent, weight);
    }
    return result;
}