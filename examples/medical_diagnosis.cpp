#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include "../include/engine.hpp"
#include "../include/types.hpp"

/**
 * Medical Diagnosis System using NLFormer
 * 
 * This example demonstrates how NLFormer can be used for medical diagnosis
 * reasoning, showing real-world application of neural logic inference.
 */

class MedicalDiagnosisSystem {
private:
    Engine engine;
    std::map<std::string, std::string> patientData;
    
public:
    MedicalDiagnosisSystem() {
        // Create medical diagnosis rules
        std::vector<Rule> medicalRules = createMedicalRules();
        engine = Engine(medicalRules);
    }
    
    std::vector<Rule> createMedicalRules() {
        std::vector<Rule> rules;
        
        // Symptom to condition rules
        rules.emplace_back(1, Pattern("has", {"?patient", "fever"}), Consequent("may_have", {"?patient", "infection"}), 0.8f);
        rules.emplace_back(2, Pattern("has", {"?patient", "cough"}), Consequent("may_have", {"?patient", "respiratory_issue"}), 0.7f);
        rules.emplace_back(3, Pattern("has", {"?patient", "headache"}), Consequent("may_have", {"?patient", "neurological_issue"}), 0.6f);
        rules.emplace_back(4, Pattern("has", {"?patient", "chest_pain"}), Consequent("may_have", {"?patient", "cardiac_issue"}), 0.9f);
        rules.emplace_back(5, Pattern("has", {"?patient", "nausea"}), Consequent("may_have", {"?patient", "digestive_issue"}), 0.5f);
        
        // Condition to diagnosis rules
        rules.emplace_back(6, Pattern("may_have", {"?patient", "infection"}), Consequent("diagnosis", {"?patient", "bacterial_infection"}), 0.6f);
        rules.emplace_back(7, Pattern("may_have", {"?patient", "respiratory_issue"}), Consequent("diagnosis", {"?patient", "pneumonia"}), 0.7f);
        rules.emplace_back(8, Pattern("may_have", {"?patient", "cardiac_issue"}), Consequent("diagnosis", {"?patient", "heart_attack"}), 0.8f);
        rules.emplace_back(9, Pattern("may_have", {"?patient", "neurological_issue"}), Consequent("diagnosis", {"?patient", "migraine"}), 0.5f);
        rules.emplace_back(10, Pattern("may_have", {"?patient", "digestive_issue"}), Consequent("diagnosis", {"?patient", "food_poisoning"}), 0.4f);
        
        // Treatment rules
        rules.emplace_back(11, Pattern("diagnosis", {"?patient", "bacterial_infection"}), Consequent("treatment", {"?patient", "antibiotics"}), 0.9f);
        rules.emplace_back(12, Pattern("diagnosis", {"?patient", "pneumonia"}), Consequent("treatment", {"?patient", "antibiotics"}), 0.8f);
        rules.emplace_back(13, Pattern("diagnosis", {"?patient", "heart_attack"}), Consequent("treatment", {"?patient", "emergency_care"}), 1.0f);
        rules.emplace_back(14, Pattern("diagnosis", {"?patient", "migraine"}), Consequent("treatment", {"?patient", "pain_relief"}), 0.7f);
        rules.emplace_back(15, Pattern("diagnosis", {"?patient", "food_poisoning"}), Consequent("treatment", {"?patient", "rest_fluids"}), 0.6f);
        
        // Age and risk factor rules
        rules.emplace_back(16, Pattern("age", {"?patient", "elderly"}), Consequent("risk_factor", {"?patient", "high_risk"}), 0.8f);
        rules.emplace_back(17, Pattern("age", {"?patient", "child"}), Consequent("risk_factor", {"?patient", "pediatric_care"}), 0.9f);
        rules.emplace_back(18, Pattern("has", {"?patient", "diabetes"}), Consequent("risk_factor", {"?patient", "complications"}), 0.7f);
        
        // Emergency rules
        rules.emplace_back(19, Pattern("diagnosis", {"?patient", "heart_attack"}), Consequent("urgency", {"?patient", "emergency"}), 1.0f);
        rules.emplace_back(20, Pattern("diagnosis", {"?patient", "bacterial_infection"}), Consequent("urgency", {"?patient", "urgent"}), 0.8f);
        rules.emplace_back(21, Pattern("diagnosis", {"?patient", "migraine"}), Consequent("urgency", {"?patient", "routine"}), 0.3f);
        
        return rules;
    }
    
    void addPatientData(const std::string& patientId, const std::string& symptom) {
        patientData[patientId] = symptom;
    }
    
    std::vector<std::pair<Consequent, float>> diagnosePatient(const std::string& patientId) {
        if (patientData.find(patientId) == patientData.end()) {
            return {};
        }
        
        std::string symptom = patientData[patientId];
        Pattern query("has", {patientId, symptom});
        
        return engine.infer(query);
    }
    
    std::vector<std::pair<Consequent, float>> comprehensiveDiagnosis(const std::string& patientId) {
        if (patientData.find(patientId) == patientData.end()) {
            return {};
        }
        
        std::string symptom = patientData[patientId];
        std::vector<Pattern> facts = {
            Pattern("has", {patientId, symptom}),
            Pattern("age", {patientId, "adult"}) // Default age
        };
        
        return engine.inferMultiLayer(facts, 3);
    }
    
    void printDiagnosis(const std::vector<std::pair<Consequent, float>>& results, const std::string& patientId) {
        std::cout << "\nMedical Diagnosis for Patient " << patientId << ":\n";
        std::cout << "=====================================\n";
        
        if (results.empty()) {
            std::cout << "No diagnosis available.\n";
            return;
        }
        
        std::cout << std::left << std::setw(25) << "Finding" << std::setw(15) << "Confidence" << "\n";
        std::cout << std::string(40, '-') << "\n";
        
        for (const auto& [consequent, confidence] : results) {
            std::string finding = "(" + consequent.predicate;
            for (const auto& arg : consequent.args) {
                finding += " " + arg;
            }
            finding += ")";
            
            std::cout << std::left << std::setw(25) << finding 
                      << std::setw(15) << std::fixed << std::setprecision(3) << confidence << "\n";
        }
    }
    
    void runDiagnosisDemo() {
        std::cout << "Medical Diagnosis System Demo\n";
        std::cout << "=============================\n";
        std::cout << "Using NLFormer for medical reasoning and diagnosis\n\n";
        
        // Test cases
        std::vector<std::pair<std::string, std::string>> testCases = {
            {"patient1", "fever"},
            {"patient2", "chest_pain"},
            {"patient3", "cough"},
            {"patient4", "headache"},
            {"patient5", "nausea"}
        };
        
        for (const auto& [patientId, symptom] : testCases) {
            addPatientData(patientId, symptom);
            
            std::cout << "\nPatient: " << patientId << " | Symptom: " << symptom;
            
            auto start = std::chrono::high_resolution_clock::now();
            auto diagnosis = comprehensiveDiagnosis(patientId);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            printDiagnosis(diagnosis, patientId);
            std::cout << "Diagnosis time: " << duration.count() << " μs\n";
        }
    }
    
    void runEmergencyTriageDemo() {
        std::cout << "\nEmergency Triage System Demo\n";
        std::cout << "============================\n";
        
        // Emergency cases
        std::vector<std::pair<std::string, std::string>> emergencyCases = {
            {"emergency1", "chest_pain"},  // Heart attack
            {"emergency2", "fever"},       // Infection
            {"emergency3", "headache"}     // Migraine
        };
        
        for (const auto& [patientId, symptom] : emergencyCases) {
            addPatientData(patientId, symptom);
            
            auto diagnosis = comprehensiveDiagnosis(patientId);
            
            // Find urgency level
            std::string urgency = "routine";
            float maxUrgency = 0.0f;
            
            for (const auto& [consequent, confidence] : diagnosis) {
                if (consequent.predicate == "urgency" && confidence > maxUrgency) {
                    urgency = consequent.args[1];
                    maxUrgency = confidence;
                }
            }
            
            std::cout << "\nPatient " << patientId << " (" << symptom << "): ";
            std::cout << "Urgency Level: " << urgency << " (confidence: " << maxUrgency << ")\n";
        }
    }
};

int main() {
    try {
        MedicalDiagnosisSystem system;
        
        system.runDiagnosisDemo();
        system.runEmergencyTriageDemo();
        
        std::cout << "\nMedical diagnosis demo completed successfully!\n";
        std::cout << "This demonstrates NLFormer's capability for real-world reasoning tasks.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
