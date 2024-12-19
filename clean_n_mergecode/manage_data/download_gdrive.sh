#!/bin/bash

# Get the Google Drive folder path from the user
folder_path="bulldog/umtn-tele-joystick/pnk/2.2/"

# Local directory to download files to
download_dir="/home/aa/DE/data_containter/gdrive_download_test/"

# Split the folder path into an array
IFS='/' read -r -a path_array <<< "$folder_path"

# Get the ID of the parent folder from the user
parent_id="1NSTkaq_nVltYc60xmP9EYMlGdp0pgcP2"

# Iterate over the directories in the path to find the final folder ID
for dir in "${path_array[@]}"; do
  # Find the folder ID
  folder_id=$(gdrive files list --query "name='$dir' and '$parent_id' in parents and mimeType='application/vnd.google-apps.folder'" | awk 'NR>1 {print $1}')

  # If the directory is not found, exit with an error
  if [[ -z "$folder_id" ]]; then
    echo "Error: Directory '$dir' not found in Google Drive."
    exit 1
  fi

  # Update the parent ID for the next iteration
  parent_id="$folder_id"
done

# Now $parent_id holds the ID of the final folder in the path

# Function to download zip files recursively
download_zip_recursive() {
  local parent_id="$1"

  # Find all zip files in the current folder
  zip_files=$(gdrive files list --query "'$parent_id' in parents and mimeType='application/zip' and trashed=false" | awk 'NR>1 {print $1}')

  # Download the zip files
  for file_id in $zip_files; do
    echo "Downloading file with ID: $file_id"
    gdrive files download --destination "$download_dir" "$file_id"
  done

  # Find all subfolders in the current folder
  subfolders=$(gdrive files list --query "'$parent_id' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false" | awk 'NR>1 {print $1}')

  # Recursively call the function for each subfolder
  for folder_id in $subfolders; do
    echo "Searching subfolder with ID: $folder_id"
    download_zip_recursive "$folder_id"
  done
}

# Start the recursive download from the specified parent folder
download_zip_recursive "$parent_id"

echo "Done!"