source=local_storage
target=remote_storage

souce_storage_mode=$source
source_data_storage=$(jq -r ".$souce_storage_mode.directory" config.json) 
source_packname=$(jq -r '.package.name' config.json)
source_packloc=$(jq -r ".$souce_storage_mode.package_loc" config.json)

target_ssh_address=$(jq -r ".$target.ssh_address" config.json)
target_storage_mode=$target
target_data_storage=$(jq -r ".$target_storage_mode.directory" config.json)
target_packname=$(jq -r '.package.name' config.json)
target_packloc=$(jq -r ".$target_storage_mode.package_loc" config.json)

# bash zip.sh $source $source_data_storage $source_packname $source_packloc

echo "Transmitting data from $source to $target"
bash transmit.sh $source $target

echo "Unzipping data at $target"
echo $target
echo $target_data_storage
echo $target_packname
echo $target_packloc
ssh $target_ssh_address "bash -s" < unzip.sh $target $target_data_storage $target_packname $target_packloc