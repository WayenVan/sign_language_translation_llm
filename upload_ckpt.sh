#!/bin/sh

REMOTE_NAME="onedrive_uni"
REMOTE_PREFIX="/Checkpoints/sign_bert"

if [ $# -eq 0 ]; then
	echo "用法: $0 文件夹1 [文件夹2 ...]"
	echo "支持 shell 通配符（glob），例如: $0 ./data/*/"
	exit 1
fi

for LOCAL_DIR in "$@"; do
	case "$LOCAL_DIR" in
	*/) LOCAL_DIR=$(echo "$LOCAL_DIR" | sed 's:/*$::') ;;
	esac

	if [ ! -d "$LOCAL_DIR" ]; then
		echo "跳过：$LOCAL_DIR 不是有效的文件夹"
		continue
	fi

	DIR_NAME=$(echo "$LOCAL_DIR" | sed 's:.*/::')
	REMOTE_PATH="${REMOTE_NAME}:${REMOTE_PREFIX}/${DIR_NAME}"

	# 检查远程目录是否存在
	rclone lsd "$REMOTE_NAME:${REMOTE_PREFIX}" | grep -F "$DIR_NAME" >/dev/null 2>&1
	if [ $? -eq 0 ]; then
		echo "⚠️ 警告：远程目录已存在，跳过上传 -> $REMOTE_PATH"
		continue
	fi

	echo "▶️ 上传 $LOCAL_DIR 到 $REMOTE_PATH ..."
	rclone copy "$LOCAL_DIR" "$REMOTE_PATH" -P -v

	if [ $? -eq 0 ]; then
		echo "✅ 成功：$DIR_NAME"
	else
		echo "❌ 失败：$DIR_NAME"
	fi
done
