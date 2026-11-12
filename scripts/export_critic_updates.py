#!/usr/bin/env python3
"""Audit recorded reward/evidence updates for all primary CEAC cells; no training."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ceac_walkthrough.mechanisms import export_reconstruction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--burden',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    export_reconstruction(args.burden,args.prior,ROOT/'notebooks/evidence/mechanisms',args.workers)

if __name__=='__main__':main()
