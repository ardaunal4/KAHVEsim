// csv_writer.h
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "gpu_update.cuh"

class CSVWriter {
public:
    static bool initializeOutputFile(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot create output file " << filename << std::endl;
            return false;
        }
        file << "time,timestep,particle_id,x,y,z,vx,vy,vz,ax,ay,az\n";
        file.close();
        std::cout << "Initialized output file: " << filename << std::endl;
        return true;
    }
    
    static bool saveTimestepSubset(const std::string& filename, const Particle* particles, int numParticles, float time, int timestep, int saveInterval = 10) {
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) return false;
        file << std::scientific << std::setprecision(6);
        for (int i = 0; i < numParticles; i += saveInterval) {
            file << time << "," << timestep << "," << i << ","
                 << particles[i].x << "," << particles[i].y << "," << particles[i].z << ","
                 << particles[i].vx << "," << particles[i].vy << "," << particles[i].vz << ","
                 << particles[i].ax << "," << particles[i].ay << "," << particles[i].az << "\n";
        }
        file.close();
        return true;
    }
    
    static bool saveStatistics(const std::string& filename, const Particle* particles, int numParticles, float totalTime, int timesteps, long long executionTime) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;
        file << "# Simulation Statistics\n";
        file << "total_particles," << numParticles << "\n";
        file << "total_timesteps," << timesteps << "\n";
        file << "simulation_time," << totalTime << "\n";
        file << "execution_time," << executionTime << "\n";
        file.close();
        std::cout << "Statistics saved to: " << filename << std::endl;
        return true;
    }
};
