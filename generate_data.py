import json
import random
from py_ballisticcalc import Ammo, Weapon, Sight, Atmo, Wind, TrajectoryCalc

def generate_example():
    try:
        ammo = Ammo(
            drag_model="G1",
            ballistic_coefficient=round(random.uniform(0.3, 0.6), 3),
            muzzle_velocity=random.randint(2400, 2900),
            weight=round(random.uniform(140, 180), 1),
            diameter=0.308
        )

        weapon = Weapon(length=random.randint(16, 26))
        sight = Sight(
            height=round(random.uniform(1.3, 2.0), 2),
            zero=random.choice([100, 150, 200])
        )

        atmo = Atmo(
            temperature=round(random.uniform(40, 90), 1),
            pressure=29.92,
            humidity=round(random.uniform(30, 80), 1),
            altitude=random.randint(0, 8000)
        )

        wind = Wind(speed=random.randint(0, 15), direction=0)

        calc = TrajectoryCalc(
            ammo=ammo,
            weapon=weapon,
            sight=sight,
            atmo=atmo,
            wind=wind
        )

        ranges = list(range(50, 1001, 50))
        results = calc.calculate(ranges)

        return {
            "input": {
                "ballistic_coefficient": ammo.ballistic_coefficient,
                "muzzle_velocity": ammo.muzzle_velocity,
                "weight": ammo.weight,
                "diameter": ammo.diameter,
                "barrel_length": weapon.length,
                "sight_height": sight.height,
                "zero_range": sight.zero,
                "temperature": atmo.temperature,
                "pressure": atmo.pressure,
                "humidity": atmo.humidity,
                "altitude": atmo.altitude,
                "wind_speed": wind.speed
            },
            "output": {
                "range_vs_drop": [
                    {"range_yd": r, "drop_in": round(results[r]["drop"], 2)}
                    for r in ranges
                ]
            }
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        return None


# Generate 1000 valid samples
examples = []
while len(examples) < 1000:
    ex = generate_example()
    if ex:
        examples.append(ex)

# Save to JSON
with open("ballistic_finetune_data_1000.json", "w") as f:
    json.dump(examples, f, indent=2)

print("✅ Generated 1000 accurate ballistic data points.")
