#!/usr/bin/env python3

from black_hole import BlackHoleSimulation, BlackHoleVisualizer
import random
import math
import numpy as np

def create_orbital_particles(simulation, num_particles=8):
    """Create particles in stable-ish orbits around the black hole"""
    bh_x, bh_y = simulation.black_hole.x, simulation.black_hole.y
    
    for i in range(num_particles):
        angle = (2 * math.pi * i / num_particles) + random.uniform(-0.3, 0.3)
        distance = random.uniform(15, 35)
        
        x = bh_x + distance * math.cos(angle)
        y = bh_y + distance * math.sin(angle)
        
        # Calculate orbital velocity for circular orbit
        orbital_speed = math.sqrt(500.0 * simulation.black_hole.mass / distance) * 0.85
        
        vx = -orbital_speed * math.sin(angle)
        vy = orbital_speed * math.cos(angle)
        
        # Add small perturbations
        vx += random.uniform(-0.3, 0.3)
        vy += random.uniform(-0.3, 0.3)
        
        mass = random.uniform(0.5, 1.5)
        simulation.add_particle(x, y, vx, vy, mass)

def create_random_particles(simulation, num_particles=6):
    """Create particles with random positions and velocities"""
    created = 0
    attempts = 0
    while created < num_particles and attempts < 50:
        attempts += 1
        
        # Create particles from edges of the simulation
        side = random.choice(['left', 'right', 'top', 'bottom'])
        
        if side == 'left':
            x, y = 0, random.uniform(20, simulation.height - 20)
            vx, vy = random.uniform(1, 4), random.uniform(-2, 2)
        elif side == 'right':
            x, y = simulation.width, random.uniform(20, simulation.height - 20)
            vx, vy = random.uniform(-4, -1), random.uniform(-2, 2)
        elif side == 'top':
            x, y = random.uniform(20, simulation.width - 20), simulation.height
            vx, vy = random.uniform(-2, 2), random.uniform(-4, -1)
        else:  # bottom
            x, y = random.uniform(20, simulation.width - 20), 0
            vx, vy = random.uniform(-2, 2), random.uniform(1, 4)
        
        distance_to_bh = math.sqrt((x - simulation.black_hole.x)**2 + 
                                  (y - simulation.black_hole.y)**2)
        if distance_to_bh > 20:
            mass = random.uniform(0.3, 1.0)
            simulation.add_particle(x, y, vx, vy, mass)
            created += 1

def create_binary_system(simulation):
    """Create a binary system that will interact with the black hole"""
    bh_x, bh_y = simulation.black_hole.x, simulation.black_hole.y
    
    # Position binary system away from black hole
    angle = random.uniform(0, 2 * math.pi)
    center_distance = 60
    center_x = bh_x + center_distance * math.cos(angle)
    center_y = bh_y + center_distance * math.sin(angle)
    
    # Create two particles orbiting each other
    separation = 3
    speed = 1.5
    
    # First particle
    x1 = center_x + separation/2
    y1 = center_y
    vx1 = 0
    vy1 = speed
    
    # Second particle
    x2 = center_x - separation/2
    y2 = center_y
    vx2 = 0
    vy2 = -speed
    
    # Add velocity toward black hole
    toward_bh_x = (bh_x - center_x) / center_distance * 0.5
    toward_bh_y = (bh_y - center_y) / center_distance * 0.5
    
    simulation.add_particle(x1, y1, vx1 + toward_bh_x, vy1 + toward_bh_y, 0.8)
    simulation.add_particle(x2, y2, vx2 + toward_bh_x, vy2 + toward_bh_y, 0.8)

def main():
    print("\n🌌 Enhanced Black Hole Simulation with Dust Physics 🌌")
    print("="*60)
    print("Features:")
    print("• Gravitational field visualization (cyan contours)")
    print("• Photon sphere (red dashed circle)")
    print("• Event horizon (orange circle)")
    print("• Accretion disk rings (gold)")
    print("• Particle trails with gradient effects")
    print("• Dynamic dust particle system")
    print("• Stellar wind effects")
    print("• Particle collisions and interactions")
    print("• Temperature-based dust coloring")
    print("• Real-time statistics display")
    print("\nControls:")
    print("• Close the window to stop")
    print("• Watch particles and dust spiral into the black hole!")
    print("="*60)
    
    simulation = BlackHoleSimulation(width=100, height=100)
    
    # Create different types of particle systems
    create_orbital_particles(simulation, num_particles=5)
    create_random_particles(simulation, num_particles=3)
    create_binary_system(simulation)
    
    # Add some initial dust for immediate visual effect
    bh_x, bh_y = simulation.black_hole.x, simulation.black_hole.y
    for i in range(50):
        angle = np.random.uniform(0, 2 * math.pi)
        distance = np.random.uniform(10, 40)
        dust_x = bh_x + distance * math.cos(angle)
        dust_y = bh_y + distance * math.sin(angle)
        
        # Random initial velocity
        dust_vx = np.random.uniform(-1, 1)
        dust_vy = np.random.uniform(-1, 1)
        
        temp = 1.0 + np.random.exponential(0.5)
        simulation.add_dust_particle(dust_x, dust_y, dust_vx, dust_vy, temperature=temp)
    
    print(f"\nInitialized {len(simulation.particles)} particles")
    print(f"Initial dust particles: {len(simulation.dust_particles)}")
    print("Starting visualization...\n")
    
    visualizer = BlackHoleVisualizer(simulation)
    
    try:
        anim = visualizer.run_animation(interval=30, frames=3000)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"Error running simulation: {e}")

if __name__ == "__main__":
    main()