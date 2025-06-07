#! /usr/bin/sh

SCRIPT_PATH="$0"
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd "$SCRIPT_DIR" || exit 1

echo "Generating stubs for Python modules..."

create_or_skip_stub() {
	if [ -d "stubs/$1" ]; then
		echo "Skipping $1, directory already exists."
	else
		echo "Creating stub for $1..."
		pyright --createstub "$1"
	fi
}

create_or_skip_stub "transformers"
create_or_skip_stub "lightning"
create_or_skip_stub "hydra"
create_or_skip_stub "mmcv"
create_or_skip_stub "mmpretrain"
create_or_skip_stub "mmpose"
create_or_skip_stub "mmseg"
create_or_skip_stub "mmdet"
create_or_skip_stub "mmengine"
create_or_skip_stub "albumentations"
create_or_skip_stub "nlpaug"
