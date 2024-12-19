#!/bin/bash

# Get the package name and location from config.json
packname=$(jq -r '.package.name' config.json)
packloc=$(jq -r ".package.location.local_storage" config.json)

# Construct the full path to the directory containing the zip files
zip_dir="$packloc/$packname"

# Get the common prefix from the file
common_prefix=$(cat common_prefix.txt)

IFS='/' read -r -a path_array <<< "$common_prefix"

parent_id="1NSTkaq_nVltYc60xmP9EYMlGdp0pgcP2"  # Start from the root of your Google Drive

# Create the directory structure in Google Drive, INCLUDING the last directory
for dir in "${path_array[@]}"; do
  # Check if the directory exists
  existing_folder_id=$(gdrive files list --query "name='$dir' and '$parent_id' in parents and mimeType='application/vnd.google-apps.folder'" | awk 'NR>1 {print $1}')

  if [[ -z "$existing_folder_id" ]]; then
    # Create the directory if it doesn't exist
    echo "Creating directory: $dir"
    folder_id=$(gdrive files mkdir "$dir" --parent "$parent_id" | awk '{print $6}')
  else
    # Use the existing directory ID
    echo "Directory exists: $dir"
    folder_id="$existing_folder_id"
  fi

  # Update the parent ID for the next iteration
  parent_id="$folder_id"
done


# Upload each zip file in the directory
find "$zip_dir" -name "*.zip" -print0 | while IFS= read -r -d '' zip_file; do
  echo "Uploading: $zip_file to $parent_id"
  gdrive files upload --parent "$parent_id" "$zip_file"
done

echo "Done!"