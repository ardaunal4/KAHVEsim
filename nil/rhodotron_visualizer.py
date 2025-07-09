import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import seaborn as sns

class ParticleDynamicsSimulator:

    def __init__(self, data_path):

        sns.set_theme()
        df = pd.read_csv(data_path, delimiter=",")
        self.data = np.zeros(len(df), dtype=[
            ('time', 'f8'),
            ('x', 'f8'),
            ('y', 'f8'),
            ('z', 'f8'),
            ('vx', 'f8'),
            ('vy', 'f8'),
            ('vz', 'f8'),
            ('synch', 'i1')
        ])
        self.data['time'] = df['time'].values
        self.data['x'] = df['x'].values
        self.data['y'] = df['y'].values
        self.data['z'] = df['z'].values
        self.data['vx'] = df['vx'].values
        self.data['vy'] = df['vy'].values
        self.data['vz'] = df['vz'].values

        self.time_stamps = np.unique(self.data['time'])
        self.len_initial_particles = len(self.data[self.data['time'] == self.time_stamps[0]])
        initial_particles = self.data[self.data['time'] == self.time_stamps[0]]
        self.data['synch'] = self.synch_particles(initial_particles, self.time_stamps)
        self.mask_list = np.ones(self.len_initial_particles, dtype = bool)
        self.eliminated_particles = 0
        self.setup_multi_plot()

    def synch_particles(self, beam, timestamps):
        synch = []
        # ADAPTED: Use adaptive range for our C++ data
        z_min, z_max = np.min(beam['z']), np.max(beam['z'])
        print(f"Z range: {z_min:.3f} to {z_max:.3f}")
        
        # Use central 20% as synch particles
        z_center = (z_min + z_max) / 2
        z_range = z_max - z_min
        z_lower = z_center - 0.1 * z_range
        z_upper = z_center + 0.1 * z_range
        
        print(f"Synch range: {z_lower:.3f} to {z_upper:.3f}")
        
        is_synch = np.array([1 if (beam['z'][i] >= z_lower and beam['z'][i] <= z_upper) 
                             else 0 for i in range(len(beam))])
        
        synch_count = np.sum(is_synch)
        print(f"Synch particles: {synch_count}/{len(beam)} ({100*synch_count/len(beam):.1f}%)")
        
        for _ in timestamps:
            synch.append(is_synch)
        synch_array = np.array(synch).reshape(len(self.data))
        return synch_array

    def setup_multi_plot(self):

        self.fig = plt.figure(figsize=(12, 8)) 
        self.fig.patch.set_facecolor('lightcyan')
        gs = self.fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])
        
        # Beam dynamics plot
        self.ax1 = self.fig.add_subplot(gs[:, 0])
        self.ax1.set_xlim(-500, 1500)
        self.ax1.set_ylim(-200, 150)
        self.ax1.set_facecolor('white')
        self.ax1.grid(True, linestyle='--', alpha=0.3)
        self.ax1.set_xlabel('z (mm)', fontsize=10)
        self.ax1.set_ylabel('y (mm)', fontsize=10)
        self.ax1.set_title('Particle Beam Trajectory', fontsize=12, pad=10)
        
        # X-Y Distribution plot
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax2.set_title('Synch Particles X-Y Distribution', fontsize=10)
        self.ax2.set_xlabel('x (mm)', fontsize=8)
        self.ax2.set_ylabel('y (mm)', fontsize=8)
        
        # X-X' Phase Space plot
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        self.ax3.set_title('Synch Particles X-X\' Phase Space', fontsize=10)
        self.ax3.set_xlabel('x (mm)', fontsize=8)
        self.ax3.set_ylabel("x' (mrad)", fontsize=8)
        
        # Y-Y' Phase Space plot 
        self.ax4 = self.fig.add_subplot(gs[2, 1])
        self.ax4.set_title('Synch Particles Y-Y\' Phase Space', fontsize=10)
        self.ax4.set_xlabel('y (mm)', fontsize=8)
        self.ax4.set_ylabel("y' (mrad)", fontsize=8)
        
        plt.tight_layout()

    def draw_boundaries(self):
        """Draw boundary lines for different regions"""
        # Region 1: -5 < z < -0.16815
        z_start, z_end = -5000, -50
        z_limits = self.ax1.get_xlim()
        total_width = z_limits[1] - z_limits[0]
        xmin = (z_start - z_limits[0]) / total_width
        xmax = (z_end - z_limits[0]) / total_width
        
        self.ax1.axhline(y=40, xmin=xmin, xmax=xmax, color='black', linewidth=1)
        self.ax1.axhline(y=-40, xmin=xmin, xmax=xmax, color='black', linewidth=1)
        
        # Region 2: -0.16815 <= z <= -0.05
        z_start, z_end = -168.15, -50
        xmin = (z_start - z_limits[0]) / total_width
        xmax = (z_end - z_limits[0]) / total_width
        
        self.ax1.axhline(y=20, xmin=xmin, xmax=xmax, color='black', linewidth=1)
        self.ax1.axhline(y=-20, xmin=xmin, xmax=xmax, color='black', linewidth=1)
        
        # Region 3: -0.05 < z < 0.5
        z_start, z_end = -49, 452
        z_points = np.array([z_start, z_end])
        y_base = 40
        tan_angle = np.tan(np.deg2rad(180-15))
        y_points = y_base + (z_points - z_start) * tan_angle
        self.ax1.plot(z_points, y_points, 'k-', linewidth=1)

        z_start, z_end = -49, 450
        z_points = np.array([z_start, z_end])
        y_base = -40
        y_points = y_base + (z_points - z_start) * tan_angle
        self.ax1.plot(z_points, y_points, 'k-', linewidth=1)
        
        # Additional boundary lines in Region 3
        z_start, z_end = 313, 430
        z_points = np.array([z_start, z_end])
        y_start, y_end = -75, -105.5
        y_points = np.array([y_start, y_end])
        self.ax1.plot(z_points, y_points, 'k-', linewidth=1)
        
        z_start, z_end = 313, 430
        z_points = np.array([z_start, z_end])
        y_start, y_end = -115, -144
        y_points = np.array([y_start, y_end])
        self.ax1.plot(z_points, y_points, 'k-', linewidth=1)
        
        # Region 4: 0.454 < z < 1.500
        z_start, z_end = 450, 1500
        xmin = (z_start - z_limits[0]) / total_width
        xmax = (z_end - z_limits[0]) / total_width
        self.ax1.axhline(y=-134 + 40, xmin=xmin, xmax=xmax, color='black', linewidth=1)

        z_start, z_end = 450, 1500
        xmin = (z_start - z_limits[0]) / total_width
        xmax = (z_end - z_limits[0]) / total_width
        self.ax1.axhline(y=-134 - 40, xmin=xmin, xmax=xmax, color='black', linewidth=1)

    def check_particle_boundaries(self, x, y, z):

        # Region 1: z < -0.5 m
        if z < -0.168:
            if abs(x) > 0.04 or abs(y) > 0.04:
                return False
                
        # Region 2: -0.168 <= z <= -0.05 m
        elif -0.168 <= z <= -0.05:
            if abs(x) > 0.04 or abs(y) > 0.02:
                return False
                
        # Region 3: -0.05 < z < 0.324 m
        elif -0.05 < z < 0.324:
            y_upper = 0.04 + (z + 0.049) * (-0.2672)
            y_lower = -0.04 + (z + 0.049) * (-0.2672)
            if abs(x) > 0.04 or y > y_upper or y < y_lower:
                return False

        # Parallel plate boundary check
        elif 0.324 <= z <= 0.438:
            y_upper = -0.075 + (z - 0.313) * (-0.2672) 
            y_lower = -0.118 + (z - 0.313) * (-0.2672)
            if abs(x) > 0.04 or y > y_upper or y < y_lower:
                return False  
 
        # Region 4: 0.438 <= z < 1.5 m
        elif 0.438 < z <= 1.5:
            if abs(x) > 0.04 or y > (-0.134 + 0.039) or y < (-0.134 - 0.040):
                return False
        
        return True
    
    def get_beam_at_time(self, time):

        return self.data[self.data['time'] == time]
    
    def calculate_statistics(self, beam):

        return {
            'n_particles': len(beam),
            'mean_x':      np.mean(beam['x']),
            'std_x':       np.std(beam['x']),
            'mean_y':      np.mean(beam['y']),
            'std_y':       np.std(beam['y']),
            'mean_vx':     np.mean(beam['vx']),
            'mean_vy':     np.mean(beam['vy'])
        }
    
    def display_statistics(self, stats):

        stats_text = [
            f"Particles:  {stats['n_particles']} / {self.len_initial_particles}({stats['n_particles']/self.len_initial_particles*100:.1f}%)",
            f"X position: {stats['mean_x']:.3f} ± {stats['std_x']:.3f} m",
            f"Y position: {stats['mean_y']:.3f} ± {stats['std_y']:.3f} m",
            f"Mean Vx:    {stats['mean_vx']:.2f}",
            f"Mean Vy:    {stats['mean_vy']:.2f}"
        ]
        
        bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
        self.ax1.text(0.02, 0.98, '\n'.join(stats_text),
                    transform=self.ax1.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=bbox_props)
    
    def display_time(self, time):
 
        time_text = f'Time: {time:.1f} ns'
        self.ax1.text(0.98, 0.98, time_text,
                    transform=self.ax1.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', 
                              boxstyle='round,pad=0.5'),
                    horizontalalignment='right',
                    verticalalignment='top')
    
    def run_simulation(self):

        self.fig.show()
        for time in self.time_stamps:

            beam = self.get_beam_at_time(time)             
            mask = []
            for i in range(len(beam)):
                is_valid = self.check_particle_boundaries(
                    beam['x'][i], 
                    beam['y'][i], 
                    beam['z'][i]
                )
                mask.append(is_valid)
            
            mask = np.array(mask, dtype=bool)
            self.mask_list = self.mask_list & mask
            beam = beam[self.mask_list]
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()

            self.ax1.set_xlim(-500, 1500)
            self.ax1.set_ylim(-200, 150)
            self.ax1.set_facecolor('white')
            self.ax1.grid(True, linestyle='--', alpha=0.3)
            self.ax1.set_xlabel('z (mm)', fontsize=10)
            self.ax1.set_ylabel('y (mm)', fontsize=10)
            self.ax1.set_title('Beam Dynamics', fontsize=12, pad=10)

            self.draw_boundaries()

            non_synch_particles = beam[beam['synch'] == 0]
            synch_particles = beam[beam['synch'] == 1]

            # Main trajectory plot
            self.ax1.scatter(non_synch_particles['z']*1000, non_synch_particles['y']*1000, 
                            s=0.2, c='blue', alpha=0.6, label='Non-Synch Particles')
            self.ax1.scatter(synch_particles['z']*1000, synch_particles['y']*1000, 
                            s=0.2, c='red', alpha=0.6, label='Synch Particles')
            self.ax1.legend(loc='lower right')

            # X-Y distribution for synch particles
            self.ax2.set_xlabel('x (mm)', fontsize=8)
            self.ax2.set_ylabel('y (mm)', fontsize=8)
            self.ax2.scatter(synch_particles['x']*1000, synch_particles['y']*1000, 
                             s=1, c='red', alpha=0.6)
            
            # Calculate x' and y' for synch particles
            x_prime = synch_particles['vx'] / synch_particles['vz']
            y_prime = synch_particles['vy'] / synch_particles['vz']
            
            # X-X' Phase Space for Synch Particles
            self.ax3.set_xlabel('x (mm)', fontsize=8)
            self.ax3.set_ylabel("x'(mrad)", fontsize=8)
            self.ax3.scatter(synch_particles['x']*1000, x_prime*1000, 
                             s=1, c='red', alpha=0.6)
            
            # Y-Y' Phase Space for Synch Particles
            self.ax4.set_xlabel('y (mm)', fontsize=8)
            self.ax4.set_ylabel("y'(mrad)", fontsize=8)
            self.ax4.scatter(synch_particles['y']*1000, y_prime*1000, 
                             s=1, c='red', alpha=0.6)

            # Statistics and time display
            stats = self.calculate_statistics(beam)
            self.display_statistics(stats)
            self.display_time(time)
            
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        
        plt.close()

if __name__ == "__main__":
    #Using our C++ simulation results
    data_path = "simulation_results.csv"
    simulation = ParticleDynamicsSimulator(data_path)
    simulation.run_simulation()