import numpy as np
import pandas as pd 
from fitparse import FitFile
import csv, datetime

ff = FitFile("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/371880531480_METRICS.fit")

# Debug: Check what message types are available
print("Available message types:")
message_types = set()
for msg in ff.get_messages():
    message_types.add(msg.name)
print(sorted(message_types))

# Debug: Check monitoring messages specifically
print("\nChecking monitoring messages...")
monitoring_count = 0
for msg in ff.get_messages("monitoring"):
    monitoring_count += 1
    if monitoring_count <= 3:  # Show first 3 messages
        print(f"Message {monitoring_count}:")
        for field in msg:
            print(f"  {field.name}: {field.value}")
        print()

print(f"Total monitoring messages: {monitoring_count}")

# Try different message types that might contain heart rate
print("\nTrying other message types for heart rate...")
for msg_type in ["record", "hr", "monitoring"]:
    print(f"\nChecking {msg_type} messages:")
    count = 0
    for msg in ff.get_messages(msg_type):
        count += 1
        if count <= 2:  # Show first 2 messages
            row = {d.name: d.value for d in msg}
            hr_fields = [k for k in row.keys() if 'heart' in k.lower() or 'hr' in k.lower()]
            if hr_fields:
                print(f"  Found HR fields: {hr_fields}")
                for field in hr_fields:
                    print(f"    {field}: {row[field]}")
        if count > 10:  # Don't check too many
            break
    print(f"  Total {msg_type} messages: {count}")

# Extract data to CSV (your existing code)
with open("daily_hr.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["timestamp_utc","heart_rate_bpm"])
    for msg in ff.get_messages("monitoring"):
        row = {d.name: d.value for d in msg}
        ts = row.get("timestamp")
        hr = row.get("heart_rate")
        if ts and hr:
            ts = ts.replace(tzinfo=datetime.timezone.utc).isoformat()
            w.writerow([ts, hr])

# Read the CSV and print head
df = pd.read_csv("daily_hr.csv")
print(df.head())