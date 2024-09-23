import os


def csv_files(folder_path):
    csv_file_list = []

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is a CSV
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                csv_file_list.append(file_path)

    return csv_file_list
