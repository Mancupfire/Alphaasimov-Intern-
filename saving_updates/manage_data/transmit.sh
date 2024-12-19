action=$1
server_type=$(jq -r '.remote_storage.server_type' config.json)
ssh_server_address=$(jq -r '.remote_storage.ssh_address' config.json)
aws_bucket=$(jq -r '.remote_storage.aws_bucket' config.json)
packname=$(jq -r '.package.name' config.json)

# ssh server
if [[ "$server_type" == "ssh" ]]; then
    if [[ "$action" == "upload" ]]; then
        source_path=$(jq -r '.package.location.local_storage' config.json)/$packname.zip
        target_dir=$(jq -r '.package.location.remote_storage' config.json)
        scp $source_path $ssh_server_address:$target_dir
    elif [[ "$action" == "download" ]]; then
        source_path=$(jq -r '.package.location.remote_storage' config.json)/$packname.zip
        target_dir=$(jq -r '.package.location.local_storage' config.json)
        scp $ssh_server_address:$source_path $target_dir
    else
        echo "Invalid action"
    fi

# aws server
elif [[ "$server_type" == "aws" ]]; then
    if [[ "$action" == "upload" ]]; then
        source_path=$(jq -r '.package.location.local_storage' config.json)/$packname.zip
        target_dir=$(jq -r '.package.location.remote_storage' config.json)
        aws $aws_bucket cp $source_path $aws_bucket:/$target_dir
    elif [[ "$action" == "download" ]]; then
        source_path=$(jq -r '.package.location.remote_storage' config.json)/$packname.zip
        target_dir=$(jq -r '.package.location.local_storage' config.json)
        echo $source_path
        exit
        aws $aws_bucket cp $aws_bucket:/$source_path $target_dir
    else
        echo "Invalid action"
    fi

else
    echo "Invalid server type"
fi

# aws server
