#!/bin/bash

# Directory containing the zip files to unzip
zip_dir="/path/to/your/zip/files"

# Directory to extract the zip files to
extract_dir="/path/to/your/extract/directory"

# Find all zip files in the specified directory
find "$zip_dir" -name "*.zip" -print0 | while IFS= read -r -d $'\0' zip_file; do
  echo "Unzipping: $zip_file"

  # Extract the zip file to the extract directory
  7z x -o"$extract_dir" "$zip_file"
done

echo "Done!"