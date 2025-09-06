import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import matplotlib.patches as patches

# Create the animation class for X vs Z with boundary conditions
class ParticleAnimationXZ:
    def __init__(self, data_path):
        print("Loading simulation data...")
        self.df = pd.read_csv(data_path)

        # Convert time to nanoseconds for better display
        self.df['time_ns'] = self.df['time'] * 1e9

        self.times = np.sort(self.df['time_ns'].unique())
        self.particle_ids = self.df['particle_id'].unique()

        # Track which particles are eliminated
        self.eliminated_particles = set()
        self.initial_particle_count = len(self.particle_ids)

        print(f"Found {len(self.times)} time steps ({self.times[0]:.1f} to {self.times[-1]:.1f} ns)")
        print(f"Tracking {self.initial_particle_count} particles")

        self.setup_plots()

    def setup_plots(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('lightcyan')
        gs = self.fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])

        # Beam dynamics plot
        self.ax1 = self.fig.add_subplot(gs[:, 0])
        self.ax1.set_xlim(-0.500, 1.500)
        self.ax1.set_ylim(-0.200, 0.150)
        self.ax1.set_facecolor('white')
        self.ax1.grid(True, linestyle='--', alpha=0.3)
        self.ax1.set_xlabel('z (m)', fontsize=10)
        self.ax1.set_ylabel('y (m)', fontsize=10)
        self.ax1.set_title('Particle Beam Trajectory', fontsize=12, pad=10)

        # X-Y Distribution plot
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax2.set_title('Synch Particles X-Y Distribution', fontsize=10)
        self.ax2.set_xlabel('x (m)', fontsize=8)
        self.ax2.set_ylabel('y (m)', fontsize=8)

        # X-X' Phase Space plot
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        self.ax3.set_title('Synch Particles X-X\' Phase Space', fontsize=10)
        self.ax3.set_xlabel('x (m)', fontsize=8)
        self.ax3.set_ylabel("x' (mrad)", fontsize=8)

        # Y-Y' Phase Space plot
        self.ax4 = self.fig.add_subplot(gs[2, 1])
        self.ax4.set_title('Synch Particles Y-Y\' Phase Space', fontsize=10)
        self.ax4.set_xlabel('y (m)', fontsize=8)
        self.ax4.set_ylabel("y'(mrad)", fontsize=8)

        plt.tight_layout()

        # Initialize scatter plots
        self.scatter1 = self.ax1.scatter([], [], s=5, c='blue', alpha=0.7, label='Active')
        self.scatter_eliminated = self.ax1.scatter([], [], s=5, c='red', alpha=0.7, label='Eliminated')
        self.scatter2 = self.ax3.scatter([], [], s=5, c='blue', alpha=0.7, label='Active') # Assuming scatter2 is for X-X' phase space
        self.scatter_xy = self.ax2.scatter([], [], s=1, c='red', alpha=0.6)
        self.scatter_xx_prime = self.ax3.scatter([], [], s=1, c='red', alpha=0.6)
        self.scatter_yy_prime = self.ax4.scatter([], [], s=1, c='red', alpha=0.6)

        # Add legend to the trajectory plot
        self.ax1.legend()

        # Text for time and statistics
        self.time_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes,
                                      bbox=dict(facecolor='white', alpha=0.8))
        self.stats_text = self.ax1.text(0.02, 0.75, '', transform=self.ax1.transAxes,
                                       bbox=dict(facecolor='white', alpha=0.8))

        # Draw boundaries
        self.draw_boundaries()

    def check_particle_boundaries(self, x, z):
        """Check if a particle is within the allowed boundaries"""
        # Convert to meters for boundary calculations
        z_m = z  # z is already in meters
        x_m = x  # x is already in meters

        # Region 1: -5 < z < -0.16815
        if -5 < z_m < -0.16815:
            if x_m > 0.040 or x_m < -0.040:
                return False

        # Region 2: -0.16815 <= z <= -0.05
        elif -0.16815 <= z_m <= -0.05:
            if x_m > 0.020 or x_m < -0.020:
                return False

        # Region 3: -0.05 < z < 0.324
        elif -0.05 < z_m < 0.324:
            x_upper = 0.04 + (z_m + 0.049) * (-0.2672)
            x_lower = -0.04 + (z_m + 0.049) * (-0.2672)
            if x_m > x_upper or x_m < x_lower:
                return False

        # Region 4: 0.324 <= z <= 0.438
        elif 0.324 <= z_m <= 0.438:
            x_upper = -0.075 + (z_m - 0.313) * (-0.2672)
            x_lower = -0.118 + (z_m - 0.313) * (-0.2672)
            if x_m > x_upper or x_m < x_lower:
                return False

        # Region 5: z >= 0.438
        elif z_m >= 0.438:
            if x_m > (-0.134 + 0.039) or x_m < (-0.134 - 0.040):
                return False

        return True

    def draw_boundaries(self):
        """Draw boundary lines for different regions"""
        # Region 1: -5 < z < -0.16815
        z_start, z_end = -5, -0.050
        z_limits = self.ax1.get_xlim()
        total_width = z_limits[1] - z_limits[0]
        xmin = (z_start - z_limits[0]) / total_width
        xmax = (z_end - z_limits[0]) / total_width

        self.ax1.axhline(y=0.040, xmin=xmin, xmax=xmax, color='black', linewidth=1)
        self.ax1.axhline(y=-0.040, xmin=xmin, xmax=xmax, color='black', linewidth=1)

        # Region 2: -0.16815 <= z <= -0.05
        z_start, z_end = -0.16815, -0.050
        xmin = (z_start - z_limits[0]) / total_width
        xmax = (z_end - z_limits[0]) / total_width

        self.ax1.axhline(y=0.020, xmin=xmin, xmax=xmax, color='black', linewidth=1)
        self.ax1.axhline(y=-0.020, xmin=xmin, xmax=xmax, color='black', linewidth=1)

        # Region 3: -0.05 < z < 0.5
        z_start, z_end = -0.049, 0.452
        z_points = np.array([z_start, z_end])
        y_base = 0.040
        tan_angle = np.tan(np.deg2rad(180-15))
        y_points = y_base + (z_points - z_start) * tan_angle
        self.ax1.plot(z_points, y_points, 'k-', linewidth=1)

        z_start, z_end = -0.049, 0.450
        z_points = np.array([z_start, z_end])
        y_base = -0.040
        y_points = y_base + (z_points - z_start) * tan_angle
        self.ax1.plot(z_points, y_points, 'k-', linewidth=1)

        # Additional boundary lines in Region 3
        z_start, z_end = 0.313, 0.430
        z_points = np.array([z_start, z_end])
        y_start, y_end = -0.075, -0.1055
        y_points = np.array([y_start, y_end])
        self.ax1.plot(z_points, y_points, 'k-', linewidth=1)

        z_start, z_end = 0.313, 0.430
        z_points = np.array([z_start, z_end])
        y_start, y_end = -0.115, -0.144
        y_points = np.array([y_start, y_end])
        self.ax1.plot(z_points, y_points, 'k-', linewidth=1)

        # Region 4: 0.454 < z < 1.500
        z_start, z_end = 0.450, 1.500
        xmin = (z_start - z_limits[0]) / total_width
        xmax = (z_end - z_limits[0]) / total_width
        self.ax1.axhline(y=-0.134 + 0.040, xmin=xmin, xmax=xmax, color='black', linewidth=1)

        z_start, z_end = 0.450, 1.500
        xmin = (z_start - z_limits[0]) / total_width
        xmax = (z_end - z_limits[0]) / total_width
        self.ax1.axhline(y=-0.134 - 0.040, xmin=xmin, xmax=xmax, color='black', linewidth=1)

    def update(self, frame):
        current_time = self.times[frame]

        # Get data for current time
        current_data = self.df[self.df['time_ns'] == current_time].copy() # Use .copy() to avoid SettingWithCopyWarning

        if len(current_data) > 0:
            # Check for eliminated particles
            active_particles = []
            eliminated_particles = []

            # Determine synch particles based on z position
            synch_z_min = -1.742
            synch_z_max = -0.668
            current_data['synch'] = ((current_data['z'] >= synch_z_min) & (current_data['z'] <= synch_z_max)).astype(int)

            non_synch_particles = current_data[current_data['synch'] == 0]
            synch_particles = current_data[current_data['synch'] == 1]

            for _, particle in current_data.iterrows():
                particle_id = particle['particle_id']

                # Skip if already eliminated
                if particle_id in self.eliminated_particles:
                    eliminated_particles.append(particle)
                    continue

                # Check boundaries
                if not self.check_particle_boundaries(particle['x'], particle['z']):
                    self.eliminated_particles.add(particle_id)
                    eliminated_particles.append(particle)
                else:
                    active_particles.append(particle)

            # Convert to DataFrames
            active_df = pd.DataFrame(active_particles) if active_particles else pd.DataFrame(columns=current_data.columns)
            eliminated_df = pd.DataFrame(eliminated_particles) if eliminated_particles else pd.DataFrame(columns=current_data.columns)

            # Update X vs Z trajectory plot
            if not active_df.empty:
                self.scatter1.set_offsets(np.column_stack([active_df['z'], active_df['x']]))
            else:
                self.scatter1.set_offsets(np.empty((0, 2)))

            if not eliminated_df.empty:
                self.scatter_eliminated.set_offsets(np.column_stack([eliminated_df['z'], eliminated_df['x']]))
            else:
                self.scatter_eliminated.set_offsets(np.empty((0, 2)))

            # Calculate angles for phase space (x' = vx/vz in mrad)
            if not active_df.empty:
                vx = active_df['vx'].values
                vz = active_df['vz'].values
                x_angle = np.where(vz != 0, (vx / vz) * 1000, 0)  # Convert to mrad

                # Update phase space plot
                self.scatter2.set_offsets(np.column_stack([active_df['x'], x_angle]))
            else:
                self.scatter2.set_offsets(np.empty((0, 2)))

            # Update text
            self.time_text.set_text(f'Time: {current_time:.1f} ns\nFrame: {frame+1}/{len(self.times)}')

            stats = (f'Active Particles: {len(active_particles)}\n'
                    f'Eliminated: {len(self.eliminated_particles)}\n'
                    f'Mean Z: {active_df["z"].mean():.3f} m\n'
                    f'Mean X: {active_df["x"].mean():.3f} m\n'
                    f'Mean Vz: {active_df["vz"].mean()/1e6:.1f} Mm/s')
            self.stats_text.set_text(stats)

            # X-Y distribution for synch particles
            if not synch_particles.empty:
                self.scatter_xy.set_offsets(np.column_stack([synch_particles['x'], synch_particles['y']]))
            else:
                self.scatter_xy.set_offsets(np.empty((0, 2)))

            # Calculate x' and y' for synch particles
            if not synch_particles.empty:
                x_prime = np.where(synch_particles['vz'] != 0, synch_particles['vx'] / synch_particles['vz'], 0)
                y_prime = np.where(synch_particles['vz'] != 0, synch_particles['vy'] / synch_particles['vz'], 0)

                # X-X' Phase Space for Synch Particles
                self.scatter_xx_prime.set_offsets(np.column_stack([synch_particles['x'], x_prime]))

                # Y-Y' Phase Space for Synch Particles
                self.scatter_yy_prime.set_offsets(np.column_stack([synch_particles['y'], y_prime]))
            else:
                self.scatter_xx_prime.set_offsets(np.empty((0, 2)))
                self.scatter_yy_prime.set_offsets(np.empty((0, 2)))

        return self.scatter1, self.scatter2, self.scatter_eliminated, self.time_text, self.stats_text, self.scatter_xy, self.scatter_xx_prime, self.scatter_yy_prime

    def create_animation(self):
        print("Creating X vs Z animation with boundary conditions...")

        # Create animation
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=min(100, len(self.times)),  # Limit to 100 frames for faster testing
            interval=100,  # 100ms between frames
            blit=True,
            repeat=True
        )

        # Convert to HTML5 video
        print("Rendering video...")
        video_html = self.ani.to_html5_video()

        # Embed the video in the notebook
        html = HTML(video_html)
        display(html)

        print("Animation complete!")
        return self.ani

# Create and display the X vs Z animation
data_path = "simulation_results.csv"
animator = ParticleAnimationXZ(data_path)
animation = animator.create_animation()