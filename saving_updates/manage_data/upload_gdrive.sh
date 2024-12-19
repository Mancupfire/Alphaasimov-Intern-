#!/bin/bash

# Get the common prefix from the file
common_prefix=$(cat common_prefix.txt)

# Get the package name and location from config.json
packname=$(jq -r '.package.name' config.json)
packloc=$(jq -r ".package.location.local_storage" config.json)

# Construct the full path to the zip file
zip_file="$packloc/$packname.zip"

IFS='/' read -r -a path_array <<< "$common_prefix"
path_array=( "${path_array[@]:0:${#path_array[@]} - 1}" )  # Remove the last element
new_common_prefix=$(IFS=/; echo "${path_array[*]}")

parent_id="1NSTkaq_nVltYc60xmP9EYMlGdp0pgcP2"  # Start from the root of your Google Drive

for dir in "${path_array[@]}"; do
  # Check if the directory exists
  echo "dir: $dir"
  existing_folder_id=$(gdrive files list --query "name='$dir' and '$parent_id' in parents and mimeType='application/vnd.google-apps.folder'" | awk 'NR>1 {print $1}')
  echo "Directory exists_______________________: $existing_folder_id"

  if [[ -z "$existing_folder_id" ]]; then
    # Create the directory if it doesn't exist
    echo "Creating directory: $dir"
    folder_id=$(gdrive files mkdir "$dir" --parent "$parent_id" | awk '{print $6}')
    #echo "'fdsfsd':   $folder_id"
  else
    # Use the existing directory ID
    echo "Directory exists: $dir"
    folder_id="$existing_folder_id"
    echo "$folder_id"
  fi

  # Update the parent ID for the next iteration
  parent_id="$folder_id"
done

# Upload the zip file to the final directory
echo "Uploading: $zip_file to $parent_id"
gdrive files upload --parent "$parent_id" "$zip_file"

echo "Done!"