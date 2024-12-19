# LOG_FILE="autodetected_wrong_data.log"
LOG_FILE="wrong_data.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file does not exist."
    exit 1
fi

while IFS=$'\t' read -r directory comment; do
    if [ -d "$directory" ]; then
        echo "Deleting directory: '$directory'"
        rm -r "$directory"  
    else
        echo "Directory not found: '$directory'. Skipping..."
    fi
done < "$LOG_FILE"

echo "Operation completed."
