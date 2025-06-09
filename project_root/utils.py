# utils.py
import csv
import os

def log_results_to_csv(filepath, tree_data_list, header=None):
    """
    Logs tree data to a CSV file.
    Args:
        filepath (str): Path to the CSV file.
        tree_data_list (list of dicts): Each dict should contain tree info.
                                        e.g., {'tree_id': id, 'world_x': x, 'world_z': z, 'dbh_cm': dbh}
        header (list of str, optional): CSV header. If None, uses keys from first dict.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    write_header = not os.path.exists(filepath) # Write header only if file is new
    if header is None and tree_data_list:
        header = list(tree_data_list[0].keys())

    with open(filepath, 'a', newline='') as csvfile: # 'a' to append
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if write_header and header is not None:
            writer.writeheader()
        for data_row in tree_data_list:
            writer.writerow(data_row)

def clear_or_init_csv(filepath, header):
    """ Clears the CSV and writes a header, or creates it with a header. """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()


if __name__ == '__main__':
    csv_path = "output_data/test_results.csv"
    csv_header = ['tree_id', 'world_x_m', 'world_z_m', 'dbh_cm']
    
    # Initialize CSV (clears previous content)
    clear_or_init_csv(csv_path, csv_header)
    
    # Example data
    data1 = [{'tree_id': 1, 'world_x_m': 0.5, 'world_z_m': 3.0, 'dbh_cm': 25.5}]
    data2 = [{'tree_id': 2, 'world_x_m': -0.5, 'world_z_m': 2.8, 'dbh_cm': 30.1}]
    
    log_results_to_csv(csv_path, data1, header=csv_header)
    log_results_to_csv(csv_path, data2, header=csv_header) # Appends
    
    print(f"CSV test data written to {csv_path}")