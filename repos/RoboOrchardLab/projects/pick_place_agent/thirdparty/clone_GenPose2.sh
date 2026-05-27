#!/bin/bash

# parameters
REPO_URL="https://github.com/Omni6DPose/GenPose2.git"
TARGET_DIR="projects/pick_place_agent/thirdparty/GenPose2"
PATCH_FILE="projects/pick_place_agent/thirdparty/patches/GenPose2.patch"

# makedir
mkdir -p "$TARGET_DIR" || {
    echo "Error: Failed to create target directory $TARGET_DIR"
    exit 1
}

# clone repo of GenPose2
if [ ! -d "$TARGET_DIR/.git" ]; then
    echo "Cloning repository..."
    git clone "$REPO_URL" "$TARGET_DIR" || {
        echo "Error: Failed to clone repository"
        exit 1
    }
else
    echo "Repository already exists, skipping clone"
fi

# apply patch
echo "Applying patch..."
git -C "$TARGET_DIR" apply --whitespace=nowarn "$PATCH_FILE" || {
    echo "Error: Failed to apply patch"
    exit 1
}

echo "Patch applied successfully!"