#include"attention.hpp"
#include<algorithm>
#include<numeric>
#include<cmath>
std::vector<float> softmax(const std::vector<float>& scores)
{
    if(scores.empty()) 
    return {};
    float maxVal=*std::max_element(scores.begin(),scores.end());
    std::vector<float> exps;
    exps.reserve(scores.size());
    for(float score : scores)
    {
        exps.push_back(std::exp(score - maxVal));
    }
    float sum=std::accumulate(exps.begin(),exps.end(),0.0f);
    std::vector<float> result;
    result.reserve(exps.size());
    for(float expVal : exps)
    {
        result.push_back(expVal/sum);
    }
    return result;
}