import pandas as pd
import os
import glob

# Change 'your_directory_path' to the directory containing your CSV files
directory_path = './'
output_file = './r_rpe_summary.csv'

# Use glob to match the pattern '*.csv' to find all csv files
csv_files = glob.glob(os.path.join(directory_path, '*/r_rpe_results/table.csv'))

# Initialize an empty dictionary to hold the data
summary_data = {}

# Loop over the list of csv files
for file in csv_files:
    # Read the csv file
    df = pd.read_csv(file)
    
    # # Extract the method names and RMSE values
    # methods = df.columns[1:]  # Assuming the first column is the file/method names
    # methods = df.transpose().columns[0:]
    # print(methods)
    methods = df.iloc[:, :1].transpose().values[0]
    rmse_values = df.iloc[:, 1:2].transpose().values[0]
    print(methods)
    print(rmse_values)
    
    # Use the filename without extension as the row label
    row_label = file.split("/")[-3]
    print(row_label)
    
    # Add the RMSE values to the summary data under the row label
    summary_data[row_label] = rmse_values
# Create a new DataFrame from the summary data
summary_df = pd.DataFrame.from_dict(summary_data, orient='index', columns=methods)

# Save the DataFrame to a new csv file
summary_df.to_csv(output_file)

print(f"Summary CSV created successfully at {output_file}.")
