#!/bin/bash

# Clear synthetic data and results script
echo "This script will clear all synthetic data and results."
echo "Current directory: $(pwd)"
echo ""

# Check if we're in the right directory
if [[ ! -d "data/synthetic_data" ]]; then
    echo "Error: Not in knowledge-incorporation directory"
    echo "Please run from: SEAL/knowledge-incorporation/"
    exit 1
fi

echo "The following directories will be cleared:"
echo "  - data/synthetic_data/train/"
echo "  - data/synthetic_data/eval/"
echo "  - data/synthetic_data/EM_SFT/"
echo "  - results/query_server/"
echo "  - results/cpt/"
echo "  - results/continual_self_edits/"
echo ""

# Show what will be deleted
echo "Files to be deleted:"
find data/synthetic_data -name "*.json" | head -5
echo "... (and more)"
echo ""

read -p "Are you sure you want to delete all synthetic data and results? (y/N): " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    echo "Clearing synthetic data..."
    rm -rf data/synthetic_data/train/*
    rm -rf data/synthetic_data/eval/*
    rm -rf data/synthetic_data/EM_SFT/*
    
    echo "Clearing results..."
    rm -rf results/query_server/*
    rm -rf results/cpt/*
    rm -rf results/continual_self_edits/*
    
    echo "Done! All synthetic data and results have been cleared."
else
    echo "Aborted. No files were deleted."
fi 