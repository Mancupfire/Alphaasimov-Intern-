#!/bin/bash

# Get the zip file path and name from config.json
zip_file_path=$(jq -r '.package.location.local_storage' config.json)
zip_file_name=$(jq -r '.package.name' config.json)
zip_file="$zip_file_path/$zip_file_name.zip"
destination_folder_id="1NSTkaq_nVltYc60xmP9EYMlGdp0pgcP2"

# Extract the zipping structure from the data directory in config.json
zipping_structure=$(jq -r '.local_storage.directory' config.json)
echo $zipping_structure

# Get the top-level folder name (e.g., "bulldog")
top_level_folder=$(echo "$zipping_structure" | cut -d '/' -f 1)
echo $top_level_folder

# Find the ID of the top-level folder in Google Drive
top_level_folder_id=$(gdrive files list --query "name='$top_level_folder' and trashed=false and mimeType='application/vnd.google-apps.folder'" | awk '{print $1}')

# If the top-level folder doesn't exist, create it
if [[ -z "$top_level_folder_id" ]]; then
  top_level_folder_id=$(gdrive files mkdir "$top_level_folder")
  echo "Created folder '$top_level_folder' in Google Drive (ID: $top_level_folder_id)"
fi

# Create the remaining folder structure in Google Drive (if it doesn't exist)
current_folder_id="$top_level_folder_id"
for dir in $(echo "$zipping_structure" | cut -d '/' -f 2- | tr '/' '\n'); do
  folder_name="$dir"
  search_result=$(gdrive files list --query "name='$folder_name' and '$current_folder_id' in parents and mimeType='application/vnd.google-apps.folder'")
  if [[ -z "$search_result" ]]; then
    # Folder doesn't exist, create it
    folder_id=$(gdrive files mkdir --parent "$current_folder_id" "$folder_name")
    echo "Created folder '$folder_name' in Google Drive (ID: $folder_id)"
  else
    # Folder exists, get its ID
    folder_id=$(echo "$search_result" | awk '{print $1}')
    echo "Found existing folder '$folder_name' in Google Drive (ID: $folder_id)"
  fi
  current_folder_id="$folder_id"
done

# Upload the zip file to the specified folder in Google Drive
gdrive files upload --parent "$destination_folder_id" "$zip_file"
echo "Uploaded '$zip_file' to Google Drive (folder ID: $destination_folder_id)"

echo "Done!"