# storage_mode=$1 # local_storage or remote_storage
# data_storage=$2 # Directory where the data is stored
# packname=$3     # Package name
# packloc=$4      # Directory where the package is stored
# packpath=$packloc/$packname # directory where the package is stored
# log_path="selected_data_list.log"

storage_mode=local_storage # data zipping should be performed locally
data_storage=$(jq -r ".$storage_mode.directory" config.json) # directory where the data is stored
packname=$(jq -r '.package.name' config.json)
# packloc=$(jq -r ".$storage_mode.package_loc" config.json)
packloc=$(jq -r ".package.location.$storage_mode" config.json)
packpath=$packloc/$packname       # directory where the package is stored
log_path="selected_data_list.log"

python select.py

mkdir -p $packpath
echo -e "\e[36m$packpath\e[0m created"

# read from the log file and copy the folders to the 'package' folder
while IFS= read -r record_dir; do
    current_folder_dir="$record_dir"
    rel_dir="${record_dir#$data_storage}" # Calculate the relative path
    record_name=$(basename "$record_dir")

    pushd . > /dev/null
    cd $record_dir

    # Find the common prefix (either 'bulldog' or 'alaska')
    common_prefix=$(echo "$record_dir" | grep -oE '(bulldog|alaska)/.*')

    # Create a temporary directory named as the common prefix
    mkdir -p "$common_prefix"

    # Copy the contents into the temporary directory (without going one level deeper)
    cp -r ./* "$common_prefix"  # Changed from '*/.' to './*'

    echo -e "\e[33mZipping $record_name ...\e[0m"

    # Zip the temporary directory, preserving the structure
    zip -r "$packpath/$record_name.zip" "$common_prefix" -v

    echo -e "\e[32m$record_name zipped!!!\e[0m"

    popd > /dev/null

    # Clean up the temporary directory
    rm -rf "$record_dir/$common_prefix"
    echo $common_prefix

    # Construct the path to the unwanted "bulldog" folder within the zip file
    first_dir=$(echo "$common_prefix" | cut -d/ -f1)
    unwanted_folder="$common_prefix/$first_dir"

    # Delete the unwanted "bulldog" folder from the zip file
    zip -d "$packpath/$record_name.zip" "$unwanted_folder/*"

    # Store the common_prefix in a temporary file
    echo "$common_prefix" > common_prefix.txt

done < "$log_path"
# zip the 'package' folder
cd $packloc
zip -re $packname.zip $packname

# delete the 'package' folder
rm -rf package
echo "Done!!!"
