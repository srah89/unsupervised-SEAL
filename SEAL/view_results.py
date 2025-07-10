#!/usr/bin/env python3
"""
Knowledge Incorporation Results Viewer
=====================================

A comprehensive tool to view and analyze SEAL knowledge incorporation experiment results.
Supports query_server, CPT, and continual_self_edits results.

Usage:
    python view_results.py summary                    # Quick overview of all results
    python view_results.py compare base iter1 iter2   # Compare experiments
    python view_results.py details path/to/result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Note: Install matplotlib, seaborn, pandas for plotting: pip install matplotlib seaborn pandas")

class ResultsViewer:
    def __init__(self, results_dir: str = "knowledge-incorporation/results"):
        self.results_dir = Path(results_dir)
        
    def detect_result_type(self, filepath: Path) -> str:
        """Detect the type of result file."""
        if "continual_self_edits" in str(filepath) and "summary" in filepath.name:
            return "continual"
        elif filepath.parent.name in ["eval", "train"] and filepath.parent.parent.name == "query_server":
            return "query_server"
        elif "cpt_" in filepath.name:
            return "cpt"
        else:
            return "unknown"
    
    def load_result(self, filepath: Path) -> Dict[str, Any]:
        """Load and validate a result file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            data['_filepath'] = str(filepath)
            data['_type'] = self.detect_result_type(filepath)
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def find_all_results(self) -> Dict[str, List[Path]]:
        """Find all result files organized by type."""
        results = {
            "query_server": [],
            "cpt": [],
            "continual": []
        }
        
        # Query server results
        qs_dir = self.results_dir / "query_server"
        if qs_dir.exists():
            results["query_server"].extend(qs_dir.glob("**/*.json"))
        
        # CPT results
        cpt_dir = self.results_dir / "cpt"
        if cpt_dir.exists():
            results["cpt"].extend(cpt_dir.glob("*.json"))
        
        # Continual self-edits results
        cont_dir = self.results_dir / "continual_self_edits"
        if cont_dir.exists():
            results["continual"].extend(cont_dir.glob("**/summary_*.json"))
        
        return results
    
    def print_summary(self):
        """Print a summary of all available results."""
        print("🔬 SEAL Knowledge Incorporation Results Summary")
        print("=" * 60)
        
        results = self.find_all_results()
        
        for result_type, files in results.items():
            print(f"\n📊 {result_type.upper().replace('_', ' ')} RESULTS:")
            if not files:
                print("   No results found")
                continue
                
            for filepath in sorted(files):
                data = self.load_result(filepath)
                if not data:
                    continue
                    
                if result_type == "query_server":
                    self._print_query_server_summary(data, filepath)
                elif result_type == "cpt":
                    self._print_cpt_summary(data, filepath)
                elif result_type == "continual":
                    self._print_continual_summary(data, filepath)
    
    def _print_query_server_summary(self, data: Dict, filepath: Path):
        """Print summary for query server results."""
        overall = data.get("overall", {})
        exp_name = data.get("exp_name", filepath.stem)
        
        baseline_acc = overall.get("baseline_mean_accuracy", 0)
        adapter_acc = overall.get("adapter_mean_accuracy", 0)
        gain = overall.get("mean_gain", 0)
        n_articles = data.get("n_articles", "?")
        
        print(f"   📄 {exp_name}")
        print(f"      Baseline: {baseline_acc:.3f} → Adapter: {adapter_acc:.3f} (Gain: +{gain:.3f})")
        print(f"      Articles: {n_articles}, Dataset: {data.get('dataset', 'N/A')}")
    
    def _print_cpt_summary(self, data: Dict, filepath: Path):
        """Print summary for CPT results."""
        overall = data.get("overall", {})
        baseline_acc = overall.get("baseline_accuracy", 0)
        adapter_acc = overall.get("adapter_accuracy", 0)
        gain = overall.get("gain", 0)
        
        print(f"   📄 {filepath.stem}")
        print(f"      Baseline: {baseline_acc:.3f} → Adapter: {adapter_acc:.3f} (Gain: +{gain:.3f})")
        print(f"      Articles: {data.get('n_articles', '?')}, Completions: {data.get('k_completions', '?')}")
    
    def _print_continual_summary(self, data: Dict, filepath: Path):
        """Print summary for continual self-edits results."""
        n_seq = data.get("n_sequences", 0)
        n_data = data.get("n_datapoints", 0)
        model = data.get("base_model", "N/A")
        
        # Extract final accuracy (bottom-right of matrix)
        mean_matrix = data.get("mean_over_sequences", [])
        if mean_matrix:
            final_acc = mean_matrix[-1][-1] if mean_matrix[-1] else 0
            initial_acc = mean_matrix[0][0] if mean_matrix[0] else 0
            improvement = final_acc - initial_acc
            
            print(f"   📄 {filepath.parent.name}")
            print(f"      Initial: {initial_acc:.3f} → Final: {final_acc:.3f} (Change: {improvement:+.3f})")
            print(f"      Sequences: {n_seq}, Datapoints: {n_data}, Model: {model}")
    
    def show_detailed_results(self, filepath: str):
        """Show detailed results for a specific file."""
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {filepath}")
            return
        
        data = self.load_result(path)
        if not data:
            return
        
        result_type = data["_type"]
        
        print(f"🔍 DETAILED RESULTS: {path.name}")
        print("=" * 60)
        
        if result_type == "query_server":
            self._show_query_server_details(data)
        elif result_type == "cpt":
            self._show_cpt_details(data)
        elif result_type == "continual":
            self._show_continual_details(data)
        else:
            print(f"Unknown result type for {filepath}")
    
    def _show_query_server_details(self, data: Dict):
        """Show detailed query server results."""
        overall = data.get("overall", {})
        
        print("📈 OVERALL METRICS:")
        for key, value in overall.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n⚙️  EXPERIMENT CONFIG:")
        print(f"   Dataset: {data.get('dataset', 'N/A')}")
        print(f"   Articles: {data.get('n_articles', 'N/A')}")
        print(f"   Completions: {data.get('k_completions', 'N/A')}")
        print(f"   Eval times: {data.get('eval_times', 'N/A')}")
        
        lora_params = data.get("lora_params", {})
        if lora_params:
            print(f"\n🔧 LORA PARAMETERS:")
            for key, value in lora_params.items():
                print(f"   {key}: {value}")
        
        # Show top performing articles
        articles = data.get("articles", [])
        if articles:
            print(f"\n🏆 TOP PERFORMING ARTICLES (by gain):")
            article_gains = [(art["title"], art["stats"]["mean_gain"]) for art in articles if "stats" in art]
            article_gains.sort(key=lambda x: x[1], reverse=True)
            
            for i, (title, gain) in enumerate(article_gains[:5]):
                print(f"   {i+1}. {title}: +{gain:.3f}")
    
    def _show_cpt_details(self, data: Dict):
        """Show detailed CPT results."""
        overall = data.get("overall", {})
        
        print("📈 OVERALL METRICS:")
        for key, value in overall.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n⚙️  EXPERIMENT CONFIG:")
        print(f"   Dataset: {data.get('dataset', 'N/A')}")
        print(f"   Articles: {data.get('n_articles', 'N/A')}")
        print(f"   Completions: {data.get('k_completions', 'N/A')}")
        
        lora_params = data.get("lora_params", {})
        if lora_params:
            print(f"\n🔧 LORA PARAMETERS:")
            for key, value in lora_params.items():
                print(f"   {key}: {value}")
    
    def _show_continual_details(self, data: Dict):
        """Show detailed continual self-edits results."""
        print("📈 EXPERIMENT CONFIG:")
        print(f"   Sequences: {data.get('n_sequences', 'N/A')}")
        print(f"   Datapoints: {data.get('n_datapoints', 'N/A')}")
        print(f"   Dataset: {data.get('dataset', 'N/A')}")
        print(f"   Base model: {data.get('base_model', 'N/A')}")
        
        mean_matrix = data.get("mean_over_sequences", [])
        std_matrix = data.get("std_over_sequences", [])
        
        if mean_matrix:
            print(f"\n📊 ACCURACY MATRIX (Mean over sequences):")
            self._print_matrix(mean_matrix, "Accuracy")
    
    def _print_matrix(self, matrix: List[List[float]], title: str):
        """Print a matrix in a readable format."""
        if not matrix:
            return
            
        print(f"   {title} Matrix:")
        n_cols = len(matrix[0]) if matrix else 0
        
        # Header
        header = "   Step\\Data " + " ".join(f"d{i:>6}" for i in range(n_cols))
        print(header)
        print("   " + "-" * len(header))
        
        for i, row in enumerate(matrix):
            step_name = "Base" if i == 0 else f"Step{i-1}"
            row_str = f"   {step_name:<9} "
            
            for j, val in enumerate(row):
                if i == 0 or j < i:  # Only show relevant cells
                    row_str += f"{val:>7.3f}"
                else:
                    row_str += f"{'':>7}"
            
            print(row_str)
    
    def compare_experiments(self, experiment_names: List[str]):
        """Compare multiple experiments."""
        print("🔄 EXPERIMENT COMPARISON")
        print("=" * 60)
        
        results = []
        
        # Try to find results for each experiment name
        for exp_name in experiment_names:
            found_files = []
            
            # Search in query_server/eval
            eval_dir = self.results_dir / "query_server" / "eval"
            if eval_dir.exists():
                found_files.extend(eval_dir.glob(f"{exp_name}.json"))
            
            # Search in cpt
            cpt_dir = self.results_dir / "cpt"
            if cpt_dir.exists():
                found_files.extend(cpt_dir.glob(f"cpt_{exp_name}.json"))
            
            if found_files:
                data = self.load_result(found_files[0])
                if data:
                    results.append((exp_name, data))
            else:
                print(f"⚠️  No results found for experiment: {exp_name}")
        
        if len(results) < 2:
            print("Need at least 2 experiments to compare")
            return
        
        # Create comparison table
        print(f"\n📊 COMPARISON TABLE:")
        print(f"{'Experiment':<15} {'Baseline':<10} {'Adapter':<10} {'Gain':<10} {'Articles':<10}")
        print("-" * 60)
        
        for exp_name, data in results:
            if data["_type"] == "query_server":
                overall = data.get("overall", {})
                baseline = overall.get("baseline_mean_accuracy", 0)
                adapter = overall.get("adapter_mean_accuracy", 0)
                gain = overall.get("mean_gain", 0)
            elif data["_type"] == "cpt":
                overall = data.get("overall", {})
                baseline = overall.get("baseline_accuracy", 0)
                adapter = overall.get("adapter_accuracy", 0)
                gain = overall.get("gain", 0)
            else:
                continue
                
            n_articles = data.get("n_articles", "?")
            
            print(f"{exp_name:<15} {baseline:<10.3f} {adapter:<10.3f} {gain:<+10.3f} {n_articles:<10}")


def main():
    parser = argparse.ArgumentParser(description="View SEAL Knowledge Incorporation Results")
    parser.add_argument("command", choices=["summary", "details", "compare"], 
                       help="Command to run")
    parser.add_argument("experiments", nargs="*", 
                       help="For compare: experiment names. For details: file path")
    parser.add_argument("--results-dir", default="knowledge-incorporation/results", 
                       help="Results directory")
    
    args = parser.parse_args()
    
    viewer = ResultsViewer(args.results_dir)
    
    if args.command == "summary":
        viewer.print_summary()
    
    elif args.command == "details":
        if not args.experiments:
            print("Please provide a file path for details command")
            return
        viewer.show_detailed_results(args.experiments[0])
    
    elif args.command == "compare":
        if len(args.experiments) < 2:
            print("Please provide at least 2 experiment names to compare")
            return
        viewer.compare_experiments(args.experiments)


if __name__ == "__main__":
    main() 