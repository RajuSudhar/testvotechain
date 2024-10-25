import csv
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite import color

folder = "Real/"  # Path to your folder
'''
# List to store the time taken for the second algorithm
time_taken_algo3 = []
image_names = []
i = 0

# Iterate over all files in the folder (same range as previously used)
for file_name in os.listdir(folder)[2000:3001]:
    if i % 10 == 0:
        print(i)
    i += 1

    # Get image path
    image_path = os.path.join(folder, file_name)

    # --- Algorithm 2 Processing ---

    # Record the start time for Algorithm 2
    start_time = time.time()

    # Simulate image processing for Algorithm 2 (replace with actual algorithm)
    img = Image.open(image_path)
    t_count, b_count = extract_feature_val(image_path)
    file = fetch_and_compare_fingerprints(image_path, t_count, b_count)   # Replace this with your actual second algorithm
    file.get_image()

    # Record the end time for Algorithm 2
    end_time = time.time()

    # Calculate the time taken for Algorithm 2
    processing_time = end_time - start_time
    time_taken_algo3.append(processing_time)

# CSV file path


# Read the existing data from the CSV file
with open(csv_file, mode='r', newline='') as file:
    reader = csv.reader(file)
    existing_data = list(reader)

# Add new column header for Algorithm 2
if len(existing_data[0]) == 3:  # If the column for Algorithm 2 is not present yet
    existing_data[0].append("Time Taken (Algo 3)")

# Add the new time taken for Algorithm 2 to each row
for i, row in enumerate(existing_data[1:], start=0):  # Skip the header row
    if i < len(time_taken_algo3):
        row.append(time_taken_algo3[i])

# Write the updated data back to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(existing_data)

print(f"Updated graph data saved to {csv_file}")

# Plot the results
csv_file = "graph_data.csv"
plt.figure(figsize=(10, 6))
plt.plot(range(len(time_taken_algo3)), time_taken_algo3, marker='o', linestyle='-', color='r')

# Label the graph
plt.title('Time Taken to Process Each Image - Algorithm 2')
plt.xlabel('Image Index')
plt.ylabel('Time Taken (seconds)')
plt.xticks(range(len(time_taken_algo3)), range(2000, 3001), rotation=90,
           fontsize=8)  # Adjust x-axis labels for image range

# Display the graph
plt.tight_layout()
plt.show()
'''
df = pd.read_csv('graph_data.csv')
#plt.plot(df['Time Taken 1'], label='Without Feature-Based Matching', linewidth=0.7)
plt.plot(df['Time Taken 2'], label='Feature-Based Similar Set Matching', color='orange', linewidth=0.5 )  # Adjust alpha here
plt.plot(df['Time Taken 3'], label='Optimized Query with Feature Sorting & Indexing', color='blue', linewidth=0.5 )
plt.legend()
plt.xlabel('Fingerprint Dataset')
plt.ylabel('Time Taken')

plt.grid(True)
plt.show()