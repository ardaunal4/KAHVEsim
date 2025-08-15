// csv_reader.h
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

struct EFieldPoint {
    float x, y, z;        // Position coordinates [m]
    float Ex, Ey, Ez;     // Electric field components [V/m]
    
    EFieldPoint() : x(0), y(0), z(0), Ex(0), Ey(0), Ez(0) {}
    EFieldPoint(float x_, float y_, float z_, float Ex_, float Ey_, float Ez_) 
        : x(x_), y(y_), z(z_), Ex(Ex_), Ey(Ey_), Ez(Ez_) {}
};

class CSVReader {
public:
    // Load electric field data from CSV file
    static std::vector<EFieldPoint> loadElectricField(const std::string& filename, int maxRows = -1);
    
    // Helper function to check if a string is a valid number
    static bool isValidNumber(const std::string& str);
    
    // Trim whitespace from string
    static std::string trim(const std::string& str);
    
private:
    // Parse a single line of CSV
    static std::vector<std::string> parseLine(const std::string& line);
    
    // Convert string to float with error checking
    static bool safeStringToFloat(const std::string& str, float& result);
};

// Implementation
std::vector<EFieldPoint> CSVReader::loadElectricField(const std::string& filename, int maxRows) {
    std::vector<EFieldPoint> fieldPoints;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return fieldPoints;
    }
    
    std::string line;
    bool isFirstLine = true;
    int rowCount = 0;
    int validRowCount = 0;
    int invalidRowCount = 0;
    
    std::cout << "Loading electric field data from: " << filename << std::endl;
    
    while (std::getline(file, line) && (maxRows == -1 || validRowCount < maxRows)) {
        rowCount++;
        
        // Skip header line
        if (isFirstLine) {
            isFirstLine = false;
            std::cout << "Header: " << line << std::endl;
            continue;
        }
        
        // Skip empty lines
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        
        // Parse CSV line
        std::vector<std::string> tokens = parseLine(line);
        
        // Expected format: "x [m]", "y [m]", "z [m]", "Ex [V/m]", "Ey [V/m]", "Ez [V/m]"
        if (tokens.size() < 6) {
            invalidRowCount++;
            continue;
        }
        
        // Convert strings to floats
        float x, y, z, Ex, Ey, Ez;
        if (safeStringToFloat(tokens[0], x) &&
            safeStringToFloat(tokens[1], y) &&
            safeStringToFloat(tokens[2], z) &&
            safeStringToFloat(tokens[3], Ex) &&
            safeStringToFloat(tokens[4], Ey) &&
            safeStringToFloat(tokens[5], Ez)) {
            
            fieldPoints.emplace_back(x, y, z, Ex, Ey, Ez);
            validRowCount++;
            
            // Progress indicator for large files
            if (validRowCount % 10000 == 0) {
                std::cout << "Loaded " << validRowCount << " field points..." << std::endl;
            }
        } else {
            invalidRowCount++;
        }
    }
    
    file.close();
    
    std::cout << "CSV loading complete:" << std::endl;
    std::cout << "  Total rows processed: " << rowCount << std::endl;
    std::cout << "  Valid field points: " << validRowCount << std::endl;
    std::cout << "  Invalid/skipped rows: " << invalidRowCount << std::endl;
    std::cout << "  Memory usage: ~" << (fieldPoints.size() * sizeof(EFieldPoint)) / (1024*1024) << " MB" << std::endl;
    
    return fieldPoints;
}

std::vector<std::string> CSVReader::parseLine(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        tokens.push_back(trim(token));
    }
    
    return tokens;
}

bool CSVReader::safeStringToFloat(const std::string& str, float& result) {
    if (str.empty() || !isValidNumber(str)) {
        return false;
    }
    
    try {
        result = std::stof(str);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool CSVReader::isValidNumber(const std::string& str) {
    if (str.empty()) return false;
    
    // Check for NaN, inf, or other invalid values
    std::string lower = str;
    for (size_t i = 0; i < lower.length(); i++) {
    if (lower[i] >= 'A' && lower[i] <= 'Z') {
        lower[i] = lower[i] + ('a' - 'A');
    }
}
    if (lower == "nan" || lower == "inf" || lower == "-inf" || 
        lower == "infinity" || lower == "-infinity") {
        return false;
    }
    
    // Check if string contains only valid number characters
    size_t start = 0;
    if (str[0] == '-' || str[0] == '+') start = 1;
    
    bool hasDecimal = false;
    bool hasE = false;
    
    for (size_t i = start; i < str.length(); i++) {
        char c = str[i];
        if ((c >= '0' && c <= '9')) continue;
        if (c == '.' && !hasDecimal && !hasE) {
            hasDecimal = true;
            continue;
        }
        if ((c == 'e' || c == 'E') && !hasE && i > start) {
            hasE = true;
            if (i + 1 < str.length() && (str[i+1] == '+' || str[i+1] == '-')) {
                i++; // Skip the sign after E
            }
            continue;
        }
        return false;
    }
    
    return true;
}

std::string CSVReader::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}