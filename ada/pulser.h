// pulser.h
#pragma once
#include <vector>
#include <cmath>
#include "csv_reader.h"
#include "gpu_update.cuh"

class Pulser {
public:
    Pulser(float riseTime, float duration, float threshold) : riseTime(riseTime), duration(duration), threshold(threshold) {
        fallStart = duration - riseTime;
    }
    
    void applyPulse(const std::vector<EFieldPoint>& fieldPoints, float time, float* outputFields, int numFieldPoints) {
        for (int i = 0; i < numFieldPoints; ++i) {
            float delay = calculateDelay(fieldPoints[i].z);
            float scale = calculatePulseAmplitude(time, delay);
            outputFields[3 * i + 0] *= scale;
            outputFields[3 * i + 1] *= scale;
            outputFields[3 * i + 2] *= scale;
        }
    }
    
    float calculatePulseAmplitude(float time, float delay) const {
        float adjustedTime = time - delay;
        if (adjustedTime < 0.0f) return 0.0f;
        if (adjustedTime < riseTime) return adjustedTime / riseTime;
        if (adjustedTime < fallStart) return 1.0f;
        if (adjustedTime < duration) return 1.0f - (adjustedTime - fallStart) / riseTime;
        return 0.0f;
    }
    
    float calculateDelay(float z) const {
        return (z < threshold) ? 0.0f : 2.5f + 3.232f;
    }
    
    void getPulseParameters(float& riseTime, float& duration, float& fallStart, float& threshold) const {
        riseTime = this->riseTime;
        duration = this->duration;
        fallStart = this->fallStart;
        threshold = this->threshold;
    }

private:
    float riseTime, duration, fallStart, threshold;
};