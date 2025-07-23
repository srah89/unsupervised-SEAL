#!/bin/bash

# SEAL Docker Management Script

set -e

show_help() {
    cat << EOF
SEAL Docker Management Script

Usage: $0 [COMMAND]

Commands:
    build       Build the Docker image
    run         Run the container interactively
    jupyter     Start Jupyter notebook server
    compose     Use docker-compose (recommended)
    clean       Clean up Docker resources
    help        Show this help

Examples:
    $0 build                    # Build the image
    $0 run                      # Run interactive container
    $0 jupyter                  # Start Jupyter server
    $0 compose                  # Use docker-compose for full setup

EOF
}

build_image() {
    echo "Building SEAL Docker image..."
    docker build -t seal-yoruba .
    echo "Image built successfully!"
}

run_container() {
    echo "Running SEAL container..."
    docker run -it --rm \
        --gpus all \
        --shm-size=8g \
        -p 8000:8000 \
        -p 8080:8080 \
        -p 8265:8265 \
        -p 11111:11111 \
        -p 11112:11112 \
        -v "$(pwd)":/workspace \
        seal-yoruba
}

run_jupyter() {
    echo "Starting Jupyter notebook server..."
    docker run -it --rm \
        --gpus all \
        --shm-size=8g \
        -p 8888:8888 \
        -v "$(pwd)":/workspace \
        seal-yoruba \
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
}

run_compose() {
    echo "Starting with docker-compose..."
    docker-compose up -d
    echo "Services started. Access:"
    echo "  - Main container: docker-compose exec seal bash"
    echo "  - Jupyter: http://localhost:8888"
    echo "  - Stop with: docker-compose down"
}

clean_resources() {
    echo "Cleaning Docker resources..."
    docker system prune -f
    docker volume prune -f
    echo "Cleanup complete!"
}

# Main script logic
case "${1:-help}" in
    build)
        build_image
        ;;
    run)
        build_image
        run_container
        ;;
    jupyter)
        build_image
        run_jupyter
        ;;
    compose)
        run_compose
        ;;
    clean)
        clean_resources
        ;;
    help|*)
        show_help
        ;;
esac 