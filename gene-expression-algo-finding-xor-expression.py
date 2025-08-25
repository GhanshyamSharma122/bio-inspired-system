# Gene Expression Programming (simplified) for XOR
# Outputs fitness tables in tabular format

import itertools

# Step 1: XOR truth table
truth_table = [
    {"A":0, "B":0, "XOR":0},
    {"A":0, "B":1, "XOR":1},
    {"A":1, "B":0, "XOR":1},
    {"A":1, "B":1, "XOR":0},
]

# Step 2: Initial population (candidate expressions)
population = [
    "A OR B",
    "A AND B",
    "NOT A",
    "B"
]

# Function to evaluate a logical expression on A,B
def eval_expr(expr, A, B):
    if expr == "A OR B":
        return A | B
    elif expr == "A AND B":
        return A & B
    elif expr == "NOT A":
        return 0 if A else 1
    elif expr == "A XOR B":
        return A ^ B
    elif expr == "B":
        return B
    elif expr == "A AND NOT B":
        return A & (0 if B else 1)
    elif expr == "NOT B":
        return 0 if B else 1
    else:
        return 0  # default

# Function to calculate fitness
def fitness_table(pop):
    print("\n| Candidate Expr    | Outputs | Matches | f(x) |")
    print("|------------------|---------|---------|-----|")
    table = []
    total_fitness = 0
    for expr in pop:
        outputs = [eval_expr(expr, row["A"], row["B"]) for row in truth_table]
        matches = sum([outputs[i]==truth_table[i]["XOR"] for i in range(4)])
        total_fitness += matches
        table.append((expr, outputs, matches))
        print(f"| {expr:<16} | {','.join(map(str,outputs)):<7} | {matches:<7} | {matches:<3} |")
    print(f"\nTotal Fitness: {total_fitness}")
    return table, total_fitness

# Step 3: Initial fitness
print("=== Initial Population Fitness ===")
fitness_table(population)

# Step 4: Simulate a simple crossover & mutation
# Let's crossover "A OR B" and "NOT A"
child1 = "A AND B"  # simplified example
child2 = "NOT B"    # simplified example

# Apply mutation on child2
child2_mutated = "A AND NOT B"

# New population after crossover & mutation
population_new = [
    "A OR B",
    "A OR B",
    child1,
    child2_mutated
]

print("\n=== Population After Crossover & Mutation ===")
fitness_table(population_new)
