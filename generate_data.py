"""
generate_data.py
-----------------
Generates synthetic e-commerce customer + order + return data.
Most customers behave normally. A small number of "rings" are planted:
groups of customers who share a device_id / address / bank_account and
who return an abnormally high fraction of what they buy (classic
return-abuse / friendly-fraud ring pattern).

Output: data/customers.csv, data/orders.csv  (also usable directly by
graph_detection.py)
"""

import random
import uuid
import csv
import os
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

N_NORMAL_CUSTOMERS = 900
N_RINGS = 6
RING_SIZE_RANGE = (4, 9)
ORDERS_PER_CUSTOMER_RANGE = (2, 12)

NORMAL_RETURN_RATE = 0.08   # ~8% of orders returned, typical honest behaviour
RING_RETURN_RATE = 0.75     # rings return ~75% of what they buy


def make_shared_pool(n):
    """A small pool of devices/addresses/bank accounts that a ring reuses."""
    return {
        "devices": [f"DVC-{uuid.uuid4().hex[:8]}" for _ in range(max(1, n // 3))],
        "addresses": [fake.address().replace("\n", ", ") for _ in range(max(1, n // 3))],
        "bank_accounts": [f"BANK-{random.randint(10**9, 10**10 - 1)}" for _ in range(max(1, n // 4))],
    }


def gen_customers():
    customers = []
    cid = 1

    # --- normal customers: mostly unique device/address/bank ---
    for _ in range(N_NORMAL_CUSTOMERS):
        customers.append({
            "customer_id": f"C{cid:05d}",
            "name": fake.name(),
            "device_id": f"DVC-{uuid.uuid4().hex[:8]}",
            "address": fake.address().replace("\n", ", "),
            "bank_account": f"BANK-{random.randint(10**9, 10**10 - 1)}",
            "signup_date": fake.date_between(start_date="-2y", end_date="-30d"),
            "is_ring_member": 0,
            "ring_id": "",
        })
        cid += 1

    # --- planted fraud rings: members share pooled devices/addresses/accounts ---
    for r in range(N_RINGS):
        ring_id = f"RING-{r+1}"
        size = random.randint(*RING_SIZE_RANGE)
        pool = make_shared_pool(size)
        for _ in range(size):
            customers.append({
                "customer_id": f"C{cid:05d}",
                "name": fake.name(),
                "device_id": random.choice(pool["devices"]),
                "address": random.choice(pool["addresses"]),
                "bank_account": random.choice(pool["bank_accounts"]),
                "signup_date": fake.date_between(start_date="-120d", end_date="-5d"),
                "is_ring_member": 1,
                "ring_id": ring_id,
            })
            cid += 1

    # --- realistic noise: a few normal households share an address (family/
    # roommates) with completely normal, low return-rate behaviour. This is
    # what makes the problem non-trivial — shared attributes alone are NOT
    # proof of fraud, so a naive "shared address => fraud" rule would create
    # false positives here. Our density + return-rate blend has to survive this.
    N_HOUSEHOLDS = 12
    household_customers = [c for c in customers if c["is_ring_member"] == 0]
    random.shuffle(household_customers)
    idx = 0
    for h in range(N_HOUSEHOLDS):
        hh_size = random.randint(2, 3)
        shared_addr = fake.address().replace("\n", ", ")
        for _ in range(hh_size):
            if idx >= len(household_customers):
                break
            household_customers[idx]["address"] = shared_addr
            idx += 1

    # --- ADVERSARIAL scenario: an "evasive ring" that deliberately avoids
    # sharing device/address/bank (they know naive fraud checks look for
    # that). They use DIFFERENT devices, addresses, and bank accounts per
    # member — so the shared-attribute graph sees them as strangers. Their
    # only tell: they were all recruited together and place their first
    # orders in a tight coordinated burst (a classic real-world ring
    # signature — coordinated timing survives even when identity signals
    # are laundered). This is what graph_detection.py alone CANNOT catch,
    # and what temporal_detection.py is built to catch.
    evasive_ring_size = 7
    burst_start = fake.date_between(start_date="-45d", end_date="-40d")
    for _ in range(evasive_ring_size):
        customers.append({
            "customer_id": f"C{cid:05d}",
            "name": fake.name(),
            "device_id": f"DVC-{uuid.uuid4().hex[:8]}",       # unique, not shared
            "address": fake.address().replace("\n", ", "),     # unique, not shared
            "bank_account": f"BANK-{random.randint(10**9, 10**10 - 1)}",  # unique
            "signup_date": burst_start,  # coordinated signup — the tell
            "is_ring_member": 1,
            "ring_id": "RING-EVASIVE",
        })
        cid += 1

    random.shuffle(customers)
    return customers


def gen_orders(customers):
    orders = []
    oid = 1
    for c in customers:
        n_orders = random.randint(*ORDERS_PER_CUSTOMER_RANGE)
        if c["ring_id"] == "RING-EVASIVE":
            return_rate = max(0.25, random.gauss(RING_RETURN_RATE, 0.12))
        elif c["is_ring_member"]:
            # some ring members are cautious/evasive and keep return rate lower
            return_rate = max(0.25, random.gauss(RING_RETURN_RATE, 0.18))
        else:
            return_rate = max(0.0, random.gauss(NORMAL_RETURN_RATE, 0.05))

        for i in range(n_orders):
            returned = 1 if random.random() < return_rate else 0
            if c["ring_id"] == "RING-EVASIVE":
                # newly-created, coordinated accounts: ALL their orders (not
                # just the first) happen after the burst signup window — a
                # brand-new account can't have an order predating its signup.
                # This is what makes "first order date" a reliable tell.
                days_after = 0 if i == 0 else random.randint(0, 40)
                order_date = c["signup_date"] + __import__("datetime").timedelta(
                    days=random.randint(0, 2) if i == 0 else 2 + days_after
                )
            else:
                order_date = fake.date_between(start_date="-180d", end_date="today")
            orders.append({
                "order_id": f"O{oid:06d}",
                "customer_id": c["customer_id"],
                "order_date": order_date,
                "amount": round(random.uniform(300, 15000), 2),
                "category": random.choice(["electronics", "apparel", "home", "beauty", "grocery"]),
                "returned": returned,
            })
            oid += 1
    return orders


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    customers = gen_customers()
    orders = gen_orders(customers)

    write_csv(os.path.join(OUT_DIR, "customers.csv"), customers)
    write_csv(os.path.join(OUT_DIR, "orders.csv"), orders)

    n_ring_members = sum(c["is_ring_member"] for c in customers)
    n_evasive = sum(1 for c in customers if c["ring_id"] == "RING-EVASIVE")
    print(f"Generated {len(customers)} customers ({n_ring_members} planted ring members: "
          f"{n_ring_members - n_evasive} in {N_RINGS} 'naive' rings + {n_evasive} in 1 evasive ring)")
    print(f"Generated {len(orders)} orders")
