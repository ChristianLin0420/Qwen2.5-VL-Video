#!/bin/bash

echo "Checking CUDA Docker image availability..."

# Function to check if Docker image exists
check_image() {
    local image=$1
    echo -n "Checking $image... "
    if docker pull --quiet "$image" > /dev/null 2>&1; then
        echo "✓ Available"
        return 0
    else
        echo "✗ Not found"
        return 1
    fi
}

# List of CUDA images to try (in order of preference)
images=(
    "nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04"
    "nvidia/cuda:12.4.1-cudnn8-devel-ubuntu22.04"
    "nvidia/cuda:12.4.0-cudnn9-devel-ubuntu22.04"
    "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04"
    "nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04"
    "nvidia/cuda:12.3.1-cudnn9-devel-ubuntu22.04"
)

echo "Testing CUDA Docker images..."
echo "=============================="

found=false
for image in "${images[@]}"; do
    if check_image "$image"; then
        found=true
        echo ""
        echo "Recommended: Use this image in your Dockerfile:"
        echo "FROM $image"
        break
    fi
done

if ! $found; then
    echo ""
    echo "Warning: None of the tested images were found."
    echo "Please check your internet connection or Docker Hub availability."
fi

echo ""
echo "To update your Dockerfile, replace the first line with the recommended image." 