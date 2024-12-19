storage_mode=local_storage  #
data_storage=$(jq -r ".$storage_mode.directory" config.json)
packname=$(jq -r '.package.name' config.json)
# packloc=$(jq -r ".$storage_mode.package_loc" config.json)
packloc=$(jq -r ".package.location.$storage_mode" config.json)
zipped_packpath=$packloc/$packname.zip
zip_filename=$(basename "$zipped_packpath")

# move the zip file to the target directory ans unzip it
mv "$zipped_packpath" "$data_storage"
unzip "$data_storage/$zip_filename" -d "$data_storage"

# accrording to text file, move the files to the correct directories
while IFS= read -r line; do
    record_name=$(echo "$line" | cut -f1) # 
    rel_dir=$(echo "$line" | cut -f2)
    record_dir=$data_storage/$rel_dir

    # move the file to the correct directory and unzip it
    # if the directory doesn't exist, create it
    mkdir -p $record_dir
    mv $data_storage/$packname/$record_name.zip $record_dir
    unzip $record_dir/$record_name.zip -d "$record_dir"

    rm $data_storage/$rel_dir/$record_name.zip
done < $data_storage/$packname/lookup-table.txt


rm $data_storage/$zip_filename
rm -rf $data_storage/$packname