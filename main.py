import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    from experiments import cross_rates
    from experiments.real_cross_rates import run_distance_sweep
    
    parser = argparse.ArgumentParser(description='FORTRESS experiments')
    parser.add_argument('--synthetic', action='store_true', help='2D Gaussian experiment')
    parser.add_argument('--real', action='store_true', help='Real embeddings experiment')
    
    args = parser.parse_args()
    
    if not any([args.synthetic, args.real]):
        parser.print_help()
        return
    
    if args.synthetic:
        cross_rates.main()
    
    if args.real:
        run_distance_sweep([0.70, 0.72, 0.74, 0.76, 0.78], display=[0.72, 0.74, 0.76])

if __name__ == '__main__':
    main()
