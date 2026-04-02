import argparse
from environment import SafeRegion
from experiments import cross_rates, trajectories, beta_sweep, risk_heatmap


def main():
    parser = argparse.ArgumentParser(description='Run experiments and generate figures.')
    parser.add_argument('-t', '--trajectories', action='store_true',
                       help='Generate trajectory visualization')
    parser.add_argument('-b', '--beta-sweep', action='store_true',
                       help='Generate beta sensitivity sweep')
    parser.add_argument('-r', '--risk-heatmap', action='store_true',
                       help='Generate risk landscape heatmap')
    parser.add_argument('-c', '--cross-rates', action='store_true',
                       help='Run crossing rates experiment')
    parser.add_argument('-a', '--all', action='store_true',
                       help='Run all experiments and generate all figures')
    
    args = parser.parse_args()
    
    if not any([args.trajectories, args.beta_sweep, args.risk_heatmap, args.cross_rates, args.all]):
        parser.print_help()
        return
    
    safe = SafeRegion(n_points=2000)
    
    if args.all or args.trajectories:
        trajectories.plot_trajectories(safe)
    
    if args.all or args.beta_sweep:
        beta_sweep.plot_beta_sweep(safe)
    
    if args.all or args.risk_heatmap:
        risk_heatmap.plot_risk_heatmap(safe)
    
    if args.all or args.cross_rates:
        cross_rates.main()


if __name__ == '__main__':
    main()