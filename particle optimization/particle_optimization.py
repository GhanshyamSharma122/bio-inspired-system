import numpy as np

# Problem parameters
T = 24  # number of hours
np.random.seed(42)  # reproducible prices
price = np.random.uniform(0.1, 0.5, size=T)  # simulate 24 hourly prices in $/kWh

battery_capacity = 10.0     # kWh
charge_limit = 2.0          # kWh per hour
efficiency = 0.95           # round-trip efficiency

# PSO parameters
n_particles = 30
w = 0.7
c1 = 1.5
c2 = 1.5
max_iter = 100

# Bounds for each time step: -charge_limit (discharge) to +charge_limit (charge)
minx = -charge_limit
maxx = charge_limit

# Objective function: minimize cost
def objective(x):
    soc = 0  # state of charge
    total_cost = 0
    for t in range(T):
        xt = x[t]
        if xt >= 0:
            energy = xt / efficiency  # charging is less efficient
        else:
            energy = xt * efficiency  # discharging is less efficient

        soc += xt  # raw energy added/removed
        if soc < 0 or soc > battery_capacity:
            return 1e6  # penalty for violating SOC bounds

        total_cost += price[t] * energy
    return total_cost

# Initialize swarm
positions = np.random.uniform(minx, maxx, size=(n_particles, T))
velocities = np.zeros_like(positions)

pbest_positions = positions.copy()
pbest_fitness = np.array([objective(pos) for pos in positions])

gbest_index = np.argmin(pbest_fitness)
gbest_position = pbest_positions[gbest_index].copy()
gbest_fitness = pbest_fitness[gbest_index]

# PSO main loop
for iter in range(max_iter):
    r1 = np.random.rand(n_particles, T)
    r2 = np.random.rand(n_particles, T)

    # Update velocity and position
    velocities = (
        w * velocities +
        c1 * r1 * (pbest_positions - positions) +
        c2 * r2 * (gbest_position - positions)
    )
    positions += velocities

    # Clip positions to valid charge/discharge limits
    positions = np.clip(positions, minx, maxx)

    # Evaluate new fitness
    fitness = np.array([objective(pos) for pos in positions])

    # Update personal bests
    for i in range(n_particles):
        if fitness[i] < pbest_fitness[i]:
            pbest_fitness[i] = fitness[i]
            pbest_positions[i] = positions[i]

    # Update global best
    best_idx = np.argmin(pbest_fitness)
    if pbest_fitness[best_idx] < gbest_fitness:
        gbest_fitness = pbest_fitness[best_idx]
        gbest_position = pbest_positions[best_idx].copy()

    print(f"Iteration {iter+1:02}: Best Cost = ${gbest_fitness:.4f}")

# Final Result
print("\nOptimal charge/discharge schedule (kWh):")
print(np.round(gbest_position, 2))
print(f"Final Total Cost: ${gbest_fitness:.4f}")
