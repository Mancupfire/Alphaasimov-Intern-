import csv

STOP_DISTANCE = 1  # meters
SLOWDOWN_DISTANCE = 4  # meters
SAFETY_TIME_COEFF = 0.5  # seconds
MAX_SPEED = 4  # meters per second

# Tính limit distance
def calculate_limit_distance(current_velocity):
    limit_distance = STOP_DISTANCE + SAFETY_TIME_COEFF * current_velocity
    print(f"Calculating Limit Distance:")
    print(f"  STOP_DISTANCE: {STOP_DISTANCE} m")
    print(f"  SAFETY_TIME_COEFF: {SAFETY_TIME_COEFF} s")
    print(f"  Limit Distance: {STOP_DISTANCE} + {SAFETY_TIME_COEFF} * {current_velocity} = {limit_distance} m")

    return limit_distance

# Tính limit velocity
def calculate_limit_velocity(obstacle_distance, limit_distance, slowdown_distance, max_speed):
    if slowdown_distance == limit_distance:
        raise ValueError("SLOWDOWN_DISTANCE cannot be equal to LIMIT_DISTANCE (division by zero).")
    if obstacle_distance <= limit_distance:
        print(f"  Obstacle Distance ({obstacle_distance}) <= Limit Distance ({limit_distance}). Stopping...")
        return 0

    numerator = obstacle_distance - limit_distance  # Phần tử của công thức
    denominator = slowdown_distance - limit_distance   # Phần mẫu của công thức
    limit_velocity = (numerator / denominator) * max_speed # Phần tử chia cho phần mẫu nhân với max_speed

    print(f"Calculating Limit Velocity:")
    print(f"  Numerator (Obstacle Distance - Limit Distance): {obstacle_distance} - {limit_distance} = {numerator}")
    print(f"  Denominator (Slowdown Distance - Limit Distance): {slowdown_distance} - {limit_distance} = {denominator}")
    print(f"  Limit Velocity: ({numerator} / {denominator}) * {max_speed} = {limit_velocity} m/s")

    return limit_velocity

test_cases = [
    {"current_velocity": 2.777, "obstacle_distance": 5},
    {"current_velocity": 3.055, "obstacle_distance": 4},
    {"current_velocity": 3.333, "obstacle_distance": 3},
    {"current_velocity": 3.611, "obstacle_distance": 2},
    {"current_velocity": 3.889, "obstacle_distance": 1},
    {"current_velocity": 4.167, "obstacle_distance": 0}
]

results = []
for i, case in enumerate(test_cases, start=1):
    print(f"\n=== Test Case {i} ===")
    current_velocity = case["current_velocity"]
    obstacle_distance = case["obstacle_distance"]

    limit_distance = calculate_limit_distance(current_velocity)
    limit_velocity = calculate_limit_velocity(obstacle_distance, limit_distance, SLOWDOWN_DISTANCE, MAX_SPEED)

    results.append({
        "Test Case": i,
        "Current Velocity (m/s)": current_velocity,
        "Obstacle Distance (m)": obstacle_distance,
        "Limit Distance (m)": limit_distance,
        "Limit Velocity (m/s)": limit_velocity
    })

csv_file_path = "./Checking_Formula/results.csv"
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to: {csv_file_path}")













