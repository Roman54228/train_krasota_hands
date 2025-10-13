"""
Utility script to download datasets from Roboflow.
"""

import argparse
from roboflow import Roboflow


def download_dataset(args):
    """Download dataset from Roboflow."""
    
    print(f"Connecting to Roboflow...")
    rf = Roboflow(api_key=args.api_key)
    
    print(f"Accessing workspace: {args.workspace}")
    workspace = rf.workspace(args.workspace)
    
    print(f"Accessing project: {args.project}")
    project = workspace.project(args.project)
    
    print(f"Accessing version: {args.version}")
    version = project.version(args.version)
    
    print(f"Downloading in format: {args.format}")
    dataset = version.download(args.format, location=args.output_dir)
    
    print(f"Dataset downloaded successfully to: {args.output_dir}")
    print(f"Dataset info: {dataset}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset from Roboflow')
    
    parser.add_argument('--api_key', type=str, required=True,
                        help='Roboflow API key')
    parser.add_argument('--workspace', type=str, required=True,
                        help='Workspace name')
    parser.add_argument('--project', type=str, required=True,
                        help='Project name')
    parser.add_argument('--version', type=int, required=True,
                        help='Dataset version number')
    parser.add_argument('--format', type=str, default='yolov11',
                        choices=['yolov11', 'yolov8', 'yolov5', 'folder'],
                        help='Download format')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='Output directory')
    
    args = parser.parse_args()
    download_dataset(args)

