import pandas as pd

df = pd.read_csv("data\laptops.csv")

# Simulating a weight column 
import random
df['weight'] = [random.uniform(1.0, 3.0) for _ in range(len(df))]  # Random weights between 1.0kg and 3.0kg

use_cases = {
    "lightweight_cheap": lambda row: row['weight'] <= 1.5 and row['price (usd)'] < 500,
    "gaming": lambda row: row['gpu_type'] != "integrated" and row['ram_memory'] >= 8 and row['price (usd)'] >= 500,
    "office_work": lambda row: row['gpu_type'] == "integrated" and 4 <= row['ram_memory'] <= 8 and row['price (usd)'] <= 700,
    "heavy_duty": lambda row: ("Core i7" in row['processor_tier'] or "Ryzen 7" in row['processor_tier']) and row['ram_memory'] >= 16 and "SSD" in row['primary_storage_type'],
    "student_laptop": lambda row: row['price (usd)'] < 800 and ("Core i5" in row['processor_tier'] or "Ryzen 5" in row['processor_tier']) and row['ram_memory'] >= 8 and row['weight'] <= 2.0,
    "video_editing": lambda row: row['gpu_type'] != "integrated" and row['ram_memory'] >= 16 and "SSD" in row['primary_storage_type'] and row['display_size'] >= 15.6 and row['resolution_width'] >= 1920,
    "design_work": lambda row: row['gpu_type'] != "integrated" and row['price (usd)'] >= 1000,
}

queries = {
    "lightweight_cheap": "I want a lightweight laptop that is cheap.",
    "gaming": "I need a laptop for gaming bro no cap.",
    "office_work": "I need a laptop for office work like reading emails.",
    "heavy_duty": "I am a CS student and need a heavy-duty laptop for engineering.",
    "student_laptop": "I am a student and need an affordable laptop for school.",
    "video_editing": "I need a laptop for video editing with a high-resolution display.",
    "design_work": "I need a laptop for design work with good color accuracy.",
}

# Create labeled pairs
data = []
for use_case, criteria in use_cases.items():
    for _, row in df.iterrows():
        match = criteria(row)  # Check if the laptop matches the use case
        query = queries[use_case]
        description = (
            f"{row['brand']} {row['Model']} with {row['processor_tier']} processor, "
            f"{row['ram_memory']}GB RAM, {row['gpu_type']} GPU, "
            f"priced at ${row['price (usd)']:.2f}, {row['display_size']}-inch display"
        )
        label = 1 if match else 0
        data.append((query, description, label))

labeled_df = pd.DataFrame(data, columns=["query", "description", "label"])
labeled_df.to_csv("enhanced_laptop_query_dataset.csv", index=False)

print("Enhanced dataset created successfully!")
