# Black Hole Simulation

A realistic black hole simulation with gravitational physics, particle dynamics, and stunning visualizations using Python.

## Screenshot

![Black Hole Simulation](screenshot.png)

_The simulation showing gravitational field lines, particle trajectories, event horizon, and accretion disk effects_

## Features

🌌 **Realistic Physics**

- Gravitational force calculations
- Event horizon and Schwarzschild radius
- Photon sphere visualization
- Relativistic effects near the black hole

🎨 **Advanced Visualization**

- Gravitational field contours
- Particle trails with gradient effects
- Real-time statistics display
- Accretion disk visualization
- Color-coded particles

⚡ **Interactive Elements**

- Multiple particle systems (orbital, random, binary)
- Real-time animation
- Performance optimized rendering

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip3 install -r requirements.txt
```

## Usage

Run the simulation:

```bash
python3 run_simulation.py
```

### Controls

- **Close window**: Stop the simulation
- **Watch**: Particles spiral into the black hole with realistic physics

## What You'll See

- **Cyan contour lines**: Gravitational field strength
- **Red dashed circle**: Photon sphere (unstable orbit boundary)
- **Orange circle**: Event horizon (point of no return)
- **Gold rings**: Accretion disk effect
- **Colored particles**: Various masses and velocities
- **Gradient trails**: Particle path history
- **Statistics panel**: Real-time simulation data

## Physics Explained

### Black Hole Components

- **Schwarzschild Radius**: The event horizon where escape velocity equals light speed
- **Photon Sphere**: The boundary where photons can orbit the black hole
- **Accretion Disk**: Matter spiraling into the black hole

### Particle Behavior

- **Orbital Particles**: Start in semi-stable orbits around the black hole
- **Random Particles**: Enter from simulation boundaries with various trajectories
- **Binary System**: Two particles orbiting each other while approaching the black hole

## Technical Details

### Files

- `black_hole.py`: Core simulation engine with physics calculations
- `run_simulation.py`: Main script with particle initialization
- `requirements.txt`: Python dependencies

### Key Classes

- `BlackHole`: Gravitational source with event horizon
- `Particle`: Individual objects affected by gravity
- `BlackHoleSimulation`: Physics engine and time stepping
- `BlackHoleVisualizer`: Matplotlib-based visualization

### Parameters You Can Modify

- Gravitational constant (for visual effect)
- Number of particles
- Simulation boundaries
- Animation speed
- Trail length

## Educational Value

This simulation demonstrates:

- General relativity concepts
- Orbital mechanics
- Gravitational interactions
- N-body physics simulations
- Scientific visualization techniques

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Performance

The simulation is optimized for real-time visualization with:

- Efficient particle updates
- Smart trail management
- Optimized rendering pipeline
- Configurable frame rates

Enjoy exploring the fascinating physics of black holes! 🚀
