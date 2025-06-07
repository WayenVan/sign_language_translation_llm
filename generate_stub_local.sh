#! /usr/bin/sh

SCRIPT_PATH="$0"
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd "$SCRIPT_DIR" || exit 1

echo "Generating stubs for my local packages..."

pyright --createstub model
pyright --createstub modules
pyright --createstub misc
