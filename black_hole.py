import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass, field
from typing import List, Tuple
import math
import colorsys

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    mass: float
    color: str = 'white'
    size: float = 20.0
    trail: List[Tuple[float, float]] = field(default_factory=list)
    speed_history: List[float] = field(default_factory=list)

class BlackHole:
    def __init__(self, x: float, y: float, mass: float, visual_radius: float = 3.0):
        self.x = x
        self.y = y
        self.mass = mass
        self.visual_radius = visual_radius
        self.event_horizon = visual_radius * 1.5
        self.photon_sphere = visual_radius * 2.5
        self.schwarzschild_radius = visual_radius * 0.8
    
    def gravitational_force(self, particle: Particle) -> Tuple[float, float]:
        dx = self.x - particle.x
        dy = self.y - particle.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < self.schwarzschild_radius:
            return 0, 0
        
        # Enhanced gravitational constant for visual effect
        G = 500.0
        force_magnitude = G * self.mass * particle.mass / (distance**2)
        
        # Add relativistic effects near black hole
        if distance < self.photon_sphere:
            force_magnitude *= 1.5
        
        force_x = force_magnitude * dx / distance
        force_y = force_magnitude * dy / distance
        
        return force_x, force_y
    
    def get_gravitational_field_strength(self, x: float, y: float) -> float:
        dx = self.x - x
        dy = self.y - y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < self.schwarzschild_radius:
            return float('inf')
        
        G = 500.0
        return G * self.mass / (distance**2)

class BlackHoleSimulation:
    def __init__(self, width: float = 100, height: float = 100):
        self.width = width
        self.height = height
        self.black_hole = BlackHole(width/2, height/2, 100.0, visual_radius=3.0)
        self.particles: List[Particle] = []
        self.time_step = 0.02
        self.max_trail_length = 100
        self.frame_count = 0
        self.consumed_particles = 0
        
    def add_particle(self, x: float, y: float, vx: float, vy: float, mass: float = 1.0, color: str = None):
        if color is None:
            # Random bright color
            hue = np.random.random()
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        size = 15 + mass * 10
        particle = Particle(x, y, vx, vy, mass, color, size)
        self.particles.append(particle)
    
    def update_particle(self, particle: Particle):
        fx, fy = self.black_hole.gravitational_force(particle)
        
        ax = fx / particle.mass
        ay = fy / particle.mass
        
        particle.vx += ax * self.time_step
        particle.vy += ay * self.time_step
        
        # Calculate speed for color effects
        speed = math.sqrt(particle.vx**2 + particle.vy**2)
        particle.speed_history.append(speed)
        if len(particle.speed_history) > 10:
            particle.speed_history.pop(0)
        
        new_x = particle.x + particle.vx * self.time_step
        new_y = particle.y + particle.vy * self.time_step
        
        particle.trail.append((particle.x, particle.y))
        if len(particle.trail) > self.max_trail_length:
            particle.trail.pop(0)
        
        particle.x = new_x
        particle.y = new_y
    
    def step(self):
        particles_to_remove = []
        self.frame_count += 1
        
        for i, particle in enumerate(self.particles):
            distance_to_bh = math.sqrt((particle.x - self.black_hole.x)**2 + 
                                     (particle.y - self.black_hole.y)**2)
            
            if distance_to_bh < self.black_hole.schwarzschild_radius:
                particles_to_remove.append(i)
                self.consumed_particles += 1
                continue
            
            if (particle.x < -10 or particle.x > self.width + 10 or 
                particle.y < -10 or particle.y > self.height + 10):
                particles_to_remove.append(i)
                continue
            
            self.update_particle(particle)
        
        for i in reversed(particles_to_remove):
            self.particles.pop(i)
    
    def get_positions(self) -> Tuple[List[float], List[float]]:
        x_positions = [p.x for p in self.particles]
        y_positions = [p.y for p in self.particles]
        return x_positions, y_positions
    
    def get_trails(self) -> List[List[Tuple[float, float]]]:
        return [p.trail for p in self.particles]

class BlackHoleVisualizer:
    def __init__(self, simulation: BlackHoleSimulation):
        self.simulation = simulation
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(-5, simulation.width + 5)
        self.ax.set_ylim(-5, simulation.height + 5)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('black')
        
        # Create gravitational field visualization
        self.create_gravitational_field()
        
        bh_x, bh_y = simulation.black_hole.x, simulation.black_hole.y
        bh_radius = simulation.black_hole.visual_radius
        
        # Photon sphere (unstable orbit)
        photon_sphere = Circle((bh_x, bh_y), simulation.black_hole.photon_sphere, 
                              fill=False, color='red', alpha=0.3, linewidth=2, linestyle='--')
        self.ax.add_patch(photon_sphere)
        
        # Event horizon
        event_horizon = Circle((bh_x, bh_y), simulation.black_hole.event_horizon, 
                              color='orange', alpha=0.5, zorder=5)
        self.ax.add_patch(event_horizon)
        
        # Black hole core
        black_hole_core = Circle((bh_x, bh_y), bh_radius, color='black', zorder=10)
        self.ax.add_patch(black_hole_core)
        
        # Accretion disk effect
        for i in range(5):
            ring_radius = simulation.black_hole.event_horizon + i * 0.5
            ring = Circle((bh_x, bh_y), ring_radius, fill=False, 
                         color='gold', alpha=0.2 - i*0.03, linewidth=1)
            self.ax.add_patch(ring)
        
        self.particle_scatter = self.ax.scatter([], [], s=[], c=[], zorder=15)
        self.trail_lines = []
        
        # Info text
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                     color='white', fontsize=10, verticalalignment='top')
    
    def create_gravitational_field(self):
        # Create a grid for gravitational field visualization
        x = np.linspace(0, self.simulation.width, 20)
        y = np.linspace(0, self.simulation.height, 20)
        X, Y = np.meshgrid(x, y)
        
        # Calculate field strength at each point
        field_strength = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                strength = self.simulation.black_hole.get_gravitational_field_strength(X[i,j], Y[i,j])
                field_strength[i,j] = min(strength, 1000)  # Cap for visualization
        
        # Create contour plot for gravitational field
        levels = np.logspace(1, 3, 10)
        contour = self.ax.contour(X, Y, field_strength, levels=levels, 
                                 colors='cyan', alpha=0.3, linewidths=0.5)
        
        return contour
        
    def animate(self, frame):
        self.simulation.step()
        
        # Update particles
        if self.simulation.particles:
            positions = np.array([(p.x, p.y) for p in self.simulation.particles])
            colors = [p.color for p in self.simulation.particles]
            sizes = [p.size for p in self.simulation.particles]
            
            self.particle_scatter.set_offsets(positions)
            self.particle_scatter.set_color(colors)
            self.particle_scatter.set_sizes(sizes)
        else:
            self.particle_scatter.set_offsets(np.empty((0, 2)))
            self.particle_scatter.set_color([])
            self.particle_scatter.set_sizes([])
        
        # Clear old trails
        for line in self.trail_lines:
            line.remove()
        self.trail_lines.clear()
        
        # Draw enhanced trails
        for particle in self.simulation.particles:
            if len(particle.trail) > 1:
                trail_x = [point[0] for point in particle.trail]
                trail_y = [point[1] for point in particle.trail]
                
                # Create gradient effect for trail
                for i in range(len(trail_x) - 1):
                    alpha = (i + 1) / len(trail_x) * 0.8
                    line, = self.ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                                        color=particle.color, alpha=alpha, linewidth=2)
                    self.trail_lines.append(line)
        
        # Update info display
        info = f"Frame: {self.simulation.frame_count}\n"
        info += f"Active Particles: {len(self.simulation.particles)}\n"
        info += f"Consumed by Black Hole: {self.simulation.consumed_particles}\n"
        info += f"Time: {self.simulation.frame_count * self.simulation.time_step:.1f}s"
        self.info_text.set_text(info)
        
        return [self.particle_scatter, self.info_text] + self.trail_lines
    
    def run_animation(self, interval: int = 50, frames: int = 1000):
        anim = animation.FuncAnimation(self.fig, self.animate, frames=frames, 
                                     interval=interval, blit=False, repeat=True)
        plt.title("Black Hole Simulation", color='white')
        plt.show()
        return anim