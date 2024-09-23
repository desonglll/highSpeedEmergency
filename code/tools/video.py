import os


def get_video_paths(folder_path):
    path_file_list = []

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is a CSV
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                path_file_list.append(file_path)

    return path_file_list
