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
    particle_type: str = 'normal'  # 'normal', 'dust', 'gas'
    lifetime: float = float('inf')
    age: float = 0.0
    trail: List[Tuple[float, float]] = field(default_factory=list)
    speed_history: List[float] = field(default_factory=list)
    temperature: float = 1.0  # For dust heating effects

@dataclass
class DustParticle:
    x: float
    y: float
    vx: float
    vy: float
    mass: float = 0.01
    size: float = 1.0
    color: str = 'orange'
    age: float = 0.0
    lifetime: float = 100.0
    temperature: float = 1.0

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
        self.dust_particles: List[DustParticle] = []
        self.time_step = 0.02
        self.max_trail_length = 100
        self.frame_count = 0
        self.consumed_particles = 0
        self.consumed_dust = 0
        self.dust_spawn_rate = 0.3  # Probability per frame to spawn dust
        self.max_dust_particles = 500
        self.enable_interactions = True
        self.collision_distance = 2.0
        
    def add_particle(self, x: float, y: float, vx: float, vy: float, mass: float = 1.0, color: str = None, particle_type: str = 'normal'):
        if color is None:
            # Random bright color
            hue = np.random.random()
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        size = 15 + mass * 10
        particle = Particle(x, y, vx, vy, mass, color, size, particle_type)
        self.particles.append(particle)
    
    def add_dust_particle(self, x: float, y: float, vx: float, vy: float, mass: float = 0.01, temperature: float = 1.0):
        # Color based on temperature (cooler = darker, hotter = brighter)
        temp_normalized = min(temperature, 3.0) / 3.0
        if temp_normalized < 0.5:
            # Cool dust: brown to orange
            hue = 0.08 + temp_normalized * 0.04  # Brown to orange hue
            saturation = 0.8
            value = 0.3 + temp_normalized * 0.4
        else:
            # Hot dust: orange to white
            hue = 0.12 - (temp_normalized - 0.5) * 0.12
            saturation = 0.8 - (temp_normalized - 0.5) * 0.8
            value = 0.7 + (temp_normalized - 0.5) * 0.3
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        size = 0.5 + temperature * 0.5
        lifetime = 50 + np.random.exponential(100)
        
        dust = DustParticle(x, y, vx, vy, mass, size, color, 0.0, lifetime, temperature)
        self.dust_particles.append(dust)
    
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
        
        # Only add trails for normal particles to reduce clutter
        if particle.particle_type == 'normal':
            particle.trail.append((particle.x, particle.y))
            if len(particle.trail) > self.max_trail_length:
                particle.trail.pop(0)
        
        particle.age += self.time_step
        particle.x = new_x
        particle.y = new_y
    
    def update_dust_particle(self, dust: DustParticle):
        # Dust particles experience same gravitational force but are lighter
        dx = self.black_hole.x - dust.x
        dy = self.black_hole.y - dust.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > self.black_hole.schwarzschild_radius:
            G = 500.0
            force_magnitude = G * self.black_hole.mass * dust.mass / (distance**2)
            
            # Heat up as dust gets closer to black hole
            if distance < self.black_hole.photon_sphere:
                dust.temperature = min(dust.temperature * 1.01, 5.0)
                force_magnitude *= 1.2
            
            ax = force_magnitude * dx / distance / dust.mass
            ay = force_magnitude * dy / distance / dust.mass
            
            dust.vx += ax * self.time_step
            dust.vy += ay * self.time_step
            
            # Add some random motion for dust swirling
            dust.vx += np.random.normal(0, 0.1) * self.time_step
            dust.vy += np.random.normal(0, 0.1) * self.time_step
        
        dust.x += dust.vx * self.time_step
        dust.y += dust.vy * self.time_step
        dust.age += self.time_step
    
    def check_particle_interactions(self):
        if not self.enable_interactions:
            return
        
        # Check collisions between particles
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                p1, p2 = self.particles[i], self.particles[j]
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < self.collision_distance:
                    # Simple elastic collision
                    total_mass = p1.mass + p2.mass
                    
                    # Exchange some momentum
                    v1x_new = (p1.vx * (p1.mass - p2.mass) + 2 * p2.mass * p2.vx) / total_mass
                    v1y_new = (p1.vy * (p1.mass - p2.mass) + 2 * p2.mass * p2.vy) / total_mass
                    v2x_new = (p2.vx * (p2.mass - p1.mass) + 2 * p1.mass * p1.vx) / total_mass
                    v2y_new = (p2.vy * (p2.mass - p1.mass) + 2 * p1.mass * p1.vy) / total_mass
                    
                    p1.vx, p1.vy = v1x_new, v1y_new
                    p2.vx, p2.vy = v2x_new, v2y_new
                    
                    # Separate particles to avoid overlap
                    separation = self.collision_distance * 1.1
                    angle = math.atan2(dy, dx)
                    p1.x = p2.x - separation * math.cos(angle)
                    p1.y = p2.y - separation * math.sin(angle)
                    
                    # Spawn dust from collision
                    for _ in range(3):
                        dust_angle = np.random.uniform(0, 2 * math.pi)
                        dust_speed = np.random.uniform(0.5, 2.0)
                        dust_x = (p1.x + p2.x) / 2 + np.random.normal(0, 0.5)
                        dust_y = (p1.y + p2.y) / 2 + np.random.normal(0, 0.5)
                        dust_vx = dust_speed * math.cos(dust_angle)
                        dust_vy = dust_speed * math.sin(dust_angle)
                        
                        self.add_dust_particle(dust_x, dust_y, dust_vx, dust_vy, 
                                             temperature=3.0)  # Hot from collision
        
        # Check particle-dust interactions (particles can capture dust)
        for particle in self.particles:
            nearby_dust = []
            for i, dust in enumerate(self.dust_particles):
                dx = dust.x - particle.x
                dy = dust.y - particle.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < particle.size * 0.2:  # Capture radius
                    nearby_dust.append(i)
            
            # Remove captured dust (particles grow slightly)
            for i in reversed(nearby_dust):
                if len(nearby_dust) < 5:  # Don't remove too much at once
                    particle.mass += self.dust_particles[i].mass * 0.1
                    particle.size = min(particle.size * 1.001, 30.0)
                    self.dust_particles.pop(i)
    
    def spawn_dust(self):
        # Spawn dust particles around existing particles and near black hole
        if len(self.dust_particles) < self.max_dust_particles and np.random.random() < self.dust_spawn_rate:
            
            # Spawn dust near existing particles (stellar wind effect)
            if self.particles and np.random.random() < 0.7:
                source_particle = np.random.choice(self.particles)
                angle = np.random.uniform(0, 2 * math.pi)
                distance = np.random.exponential(3.0) + 1.0
                
                dust_x = source_particle.x + distance * math.cos(angle)
                dust_y = source_particle.y + distance * math.sin(angle)
                
                # Initial velocity with some randomness
                base_speed = 0.5
                dust_vx = source_particle.vx * 0.8 + np.random.normal(0, base_speed)
                dust_vy = source_particle.vy * 0.8 + np.random.normal(0, base_speed)
                
                temp = 1.0 + np.random.exponential(0.5)
                
            # Spawn dust in accretion disk around black hole
            else:
                angle = np.random.uniform(0, 2 * math.pi)
                # Distance in accretion disk range
                disk_inner = self.black_hole.event_horizon * 1.2
                disk_outer = self.black_hole.photon_sphere * 1.5
                distance = np.random.uniform(disk_inner, disk_outer)
                
                dust_x = self.black_hole.x + distance * math.cos(angle)
                dust_y = self.black_hole.y + distance * math.sin(angle)
                
                # Orbital velocity with inward spiral
                orbital_speed = math.sqrt(500.0 * self.black_hole.mass / distance) * 0.9
                dust_vx = -orbital_speed * math.sin(angle) + np.random.normal(0, 0.2)
                dust_vy = orbital_speed * math.cos(angle) + np.random.normal(0, 0.2)
                
                # Add inward drift
                inward_speed = 0.1
                dust_vx += inward_speed * math.cos(angle + math.pi)
                dust_vy += inward_speed * math.sin(angle + math.pi)
                
                temp = 2.0 + distance / disk_outer
            
            # Check bounds
            if (0 <= dust_x <= self.width and 0 <= dust_y <= self.height):
                self.add_dust_particle(dust_x, dust_y, dust_vx, dust_vy, temperature=temp)
    
    def step(self):
        particles_to_remove = []
        dust_to_remove = []
        self.frame_count += 1
        
        # Spawn new dust particles
        self.spawn_dust()
        
        # Check interactions between particles
        self.check_particle_interactions()
        
        # Update regular particles
        for i, particle in enumerate(self.particles):
            distance_to_bh = math.sqrt((particle.x - self.black_hole.x)**2 + 
                                     (particle.y - self.black_hole.y)**2)
            
            if distance_to_bh < self.black_hole.schwarzschild_radius:
                particles_to_remove.append(i)
                self.consumed_particles += 1
                # Spawn hot dust from consumed particle
                for _ in range(5):
                    angle = np.random.uniform(0, 2 * math.pi)
                    speed = np.random.uniform(1, 3)
                    dust_x = particle.x + np.random.normal(0, 1)
                    dust_y = particle.y + np.random.normal(0, 1)
                    dust_vx = speed * math.cos(angle)
                    dust_vy = speed * math.sin(angle)
                    self.add_dust_particle(dust_x, dust_y, dust_vx, dust_vy, temperature=4.0)
                continue
            
            if (particle.x < -10 or particle.x > self.width + 10 or 
                particle.y < -10 or particle.y > self.height + 10):
                particles_to_remove.append(i)
                continue
            
            self.update_particle(particle)
        
        # Update dust particles
        for i, dust in enumerate(self.dust_particles):
            distance_to_bh = math.sqrt((dust.x - self.black_hole.x)**2 + 
                                     (dust.y - self.black_hole.y)**2)
            
            # Dust consumed by black hole
            if distance_to_bh < self.black_hole.schwarzschild_radius:
                dust_to_remove.append(i)
                self.consumed_dust += 1
                continue
            
            # Remove old or out-of-bounds dust
            if (dust.age > dust.lifetime or 
                dust.x < -20 or dust.x > self.width + 20 or 
                dust.y < -20 or dust.y > self.height + 20):
                dust_to_remove.append(i)
                continue
            
            self.update_dust_particle(dust)
        
        # Remove particles
        for i in reversed(particles_to_remove):
            self.particles.pop(i)
        
        for i in reversed(dust_to_remove):
            self.dust_particles.pop(i)
    
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
        self.dust_scatter = self.ax.scatter([], [], s=[], c=[], alpha=0.6, zorder=12)
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
        
        # Update regular particles
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
        
        # Update dust particles
        if self.simulation.dust_particles:
            dust_positions = np.array([(d.x, d.y) for d in self.simulation.dust_particles])
            dust_colors = [d.color for d in self.simulation.dust_particles]
            dust_sizes = [d.size for d in self.simulation.dust_particles]
            
            self.dust_scatter.set_offsets(dust_positions)
            self.dust_scatter.set_color(dust_colors)
            self.dust_scatter.set_sizes(dust_sizes)
        else:
            self.dust_scatter.set_offsets(np.empty((0, 2)))
            self.dust_scatter.set_color([])
            self.dust_scatter.set_sizes([])
        
        # Clear old trails
        for line in self.trail_lines:
            line.remove()
        self.trail_lines.clear()
        
        # Draw enhanced trails for regular particles only
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
        info += f"Particles: {len(self.simulation.particles)}\n"
        info += f"Dust: {len(self.simulation.dust_particles)}\n"
        info += f"Consumed: {self.simulation.consumed_particles} particles, {self.simulation.consumed_dust} dust\n"
        info += f"Time: {self.simulation.frame_count * self.simulation.time_step:.1f}s"
        self.info_text.set_text(info)
        
        return [self.particle_scatter, self.dust_scatter, self.info_text] + self.trail_lines
    
    def run_animation(self, interval: int = 50, frames: int = 1000):
        anim = animation.FuncAnimation(self.fig, self.animate, frames=frames, 
                                     interval=interval, blit=False, repeat=True)
        plt.title("Black Hole Simulation", color='white')
        plt.show()
        return anim