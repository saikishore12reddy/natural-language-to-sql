import asyncio
from core.database import seed_table

# Mock data (without hardcoded IDs, relying on SQLite auto-increment)
customers = [
    {"name": "John Doe", "city": "Bangalore"},
    {"name": "Jane Smith", "city": "Bangalore"},
    {"name": "Rahul Kumar", "city": "Bangalore"},
    {"name": "Priya Singh", "city": "Mumbai"}
]

loans = [
    {"customer_id": 1, "amount": 2500.0, "status": "APPROVED"},
    {"customer_id": 1, "amount": 500.0, "status": "PENDING"},
    {"customer_id": 2, "amount": 3500.0, "status": "APPROVED"},
    {"customer_id": 3, "amount": 1500.0, "status": "REJECTED"},
    {"customer_id": 3, "amount": 4200.0, "status": "APPROVED"}
]

print("Seeding customers...")
res1 = seed_table("customers", customers)
print(res1)

print("Seeding loans...")
res2 = seed_table("loans", loans)
print(res2)

print("\nDatabase seeded successfully!")
