#!/usr/bin/env python3
"""
correlated_churn_study.py

Correlated churn experiment for Lean Gossip paper rebuttal.

This script addresses Reviewer 3's concern (W3) about correlated churn:
"Correlated churn would simultaneously damage both the eager mesh and the 
gossip recovery chain that Lean Gossip relies on."

We simulate regional outages where a fraction of nodes in one geographic 
region go offline simultaneously for a fixed window, then measure recovery 
effectiveness across configurations.

Experiment Design:
- Baseline: Normal bimodal churn (20% churny fraction)
- Regional outage: At tick T, X% of region R goes offline for Y seconds
- Measure: Delivery rate, latency, bandwidth during and after outage

Outputs:
- correlated_churn_results.csv: Metrics for each configuration under each scenario
"""

import sys
import csv
import random
import numpy as np
import os
import io
import contextlib
from collections import defaultdict
from meshsub import MeshSubComplete


class CorrelatedChurnSimulator(MeshSubComplete):
    """
    Extended MeshSubComplete with support for correlated regional outages.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract our custom parameters before passing to parent
        self.outage_config = kwargs.pop('outage_config', None)
        super().__init__(*args, **kwargs)
        
        # Track outage state
        self.outage_active = False
        self.outage_affected_peers = set()
        self.outage_start_tick = None
        self.outage_end_tick = None
        
        # Additional metrics for outage analysis
        self.pre_outage_deliveries = 0
        self.during_outage_deliveries = 0
        self.post_outage_deliveries = 0
        self.messages_during_outage = 0
    
    def trigger_regional_outage(self, region: str, fraction: float, duration_ticks: int):
        """
        Trigger a correlated outage affecting a fraction of peers in a region.
        
        Args:
            region: Region name (e.g., 'EU', 'US', 'Asia')
            fraction: Fraction of region's peers to take offline (0.0-1.0)
            duration_ticks: How long the outage lasts
        """
        # Find all peers in the target region
        region_peers = [pid for pid, p in self.peers.items() 
                       if p.region == region and self.online[pid]]
        
        # Select fraction to take offline
        num_to_offline = int(len(region_peers) * fraction)
        random.shuffle(region_peers)
        affected = region_peers[:num_to_offline]
        
        # Take them offline
        for pid in affected:
            self.online[pid] = False
            p = self.peers[pid]
            p.is_online = False
            p.mesh_eager.clear()
            p.mesh_lazy.clear()
            self.churn_events.append({
                'tick': self.tick,
                'peer': pid,
                'type': 'correlated_outage',
                'event': 'leave',
                'region': region
            })
        
        self.outage_active = True
        self.outage_affected_peers = set(affected)
        self.outage_start_tick = self.tick
        self.outage_end_tick = self.tick + duration_ticks
        
        return len(affected)
    
    def end_regional_outage(self):
        """
        End the regional outage, bringing affected peers back online.
        """
        for pid in self.outage_affected_peers:
            self.online[pid] = True
            p = self.peers[pid]
            p.is_online = True
            p.seen.clear()  # Clear seen set on rejoin
            
            # Rebuild mesh from peer table
            cands = [n for n in p.peer_table if self.online.get(n, False)]
            random.shuffle(cands)
            p.mesh_eager = set(cands[:min(len(cands), p.D)])
            rest = [n for n in cands if n not in p.mesh_eager]
            p.mesh_lazy = set(rest[:min(len(rest), self.D_lazy)])
            
            self.churn_events.append({
                'tick': self.tick,
                'peer': pid,
                'type': 'correlated_outage',
                'event': 'rejoin',
                'region': p.region
            })
        
        self.outage_active = False
        self.outage_affected_peers.clear()
    
    def run_with_outage(self, n_ticks: int, warmup_ticks: int, 
                        outage_tick: int, outage_region: str,
                        outage_fraction: float, outage_duration_ticks: int,
                        verbose: bool = False):
        """
        Run simulation with a scheduled regional outage.
        
        Args:
            n_ticks: Total simulation ticks (excluding warmup)
            warmup_ticks: Warmup period
            outage_tick: Tick (relative to measurement start) when outage begins
            outage_region: Which region experiences the outage
            outage_fraction: Fraction of region to take offline
            outage_duration_ticks: Duration of outage in ticks
        """
        # Run warmup using parent's step method
        if warmup_ticks > 0:
            if verbose:
                print(f"Running warmup for {warmup_ticks} ticks...")
            for _ in range(warmup_ticks):
                self.step()
            
            # Reset metrics after warmup
            self._reset_metrics()
            self.tick = 0
            self.warmup_complete = True
            if verbose:
                print(f"Warmup complete, starting measurement")
        
        # Measurement period with outage injection
        for t in range(n_ticks):
            # Check for outage trigger
            if t == outage_tick and not self.outage_active and outage_region != 'none':
                num_affected = self.trigger_regional_outage(
                    outage_region, outage_fraction, outage_duration_ticks)
                if verbose:
                    print(f"[Tick {t}] OUTAGE START: {num_affected} peers in {outage_region} went offline")
            
            # Check for outage end
            if self.outage_active and self.tick >= self.outage_end_tick:
                self.end_regional_outage()
                if verbose:
                    print(f"[Tick {t}] OUTAGE END: Affected peers back online")
            
            # Normal simulation step
            self.step()
            
            if verbose and t % 1000 == 0:
                online = sum(self.online.values())
                print(f"[Tick {t}] Online: {online}/{self.num_peers}, "
                      f"Outage active: {self.outage_active}")
    
    def _reset_metrics(self):
        """Reset metrics at end of warmup"""
        for p in self.peers.values():
            p.bytes_sent_payload = 0
            p.bytes_sent_control = 0
            p.bytes_sent_meshmnt = 0
            p.bytes_sent_gossip = 0
        self.drops_peer = 0
        self.drops_offline = 0
        self.churn_events.clear()
        self.produced_messages.clear()
        self.latency_tracker.clear()
        self.delivered.clear()


def run_correlated_churn_experiment(
    D: int,
    D_lazy: int,
    churny_fraction: float,
    outage_region: str,
    outage_fraction: float,
    outage_duration_sec: float,
    seed: int = 42
) -> dict:
    """
    Run a single correlated churn experiment.
    
    Args:
        D: Eager mesh degree
        D_lazy: Gossip degree
        churny_fraction: Background churn level
        outage_region: Region to hit with outage ('EU', 'US', 'Asia')
        outage_fraction: Fraction of region to take offline (e.g., 0.5 = 50%)
        outage_duration_sec: Duration of outage in seconds
        seed: Random seed
    
    Returns:
        Dict with metrics
    """
    tick_sec = 0.1
    outage_duration_ticks = int(outage_duration_sec / tick_sec)
    
    # Outage happens at tick 1000 (100 seconds into measurement)
    # This gives time for mesh to stabilize, then we measure recovery
    outage_tick = 1000
    
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CorrelatedChurnSimulator(
            num_peers=1000,
            tick_sec=tick_sec,
            heartbeat_interval_sec=1.0,
            peer_table_min=40,
            peer_table_max=80,
            D=D,
            D_lazy=D_lazy,
            validators_per_node=1,
            global_bandwidth_bps=25_000_000,
            msg_size_default=1536,
            mcache_gossip=3,
            churny_fraction=churny_fraction,
            churny_leave_rate=0.01,
            churny_rejoin_rate=0.05,
            region_names=['EU', 'US', 'Asia'],
            region_fractions=[0.3, 0.4, 0.3],
            seed=seed,
        )
        
        # Run with outage
        sim.run_with_outage(
            n_ticks=5000,
            warmup_ticks=1000,
            outage_tick=outage_tick,
            outage_region=outage_region,
            outage_fraction=outage_fraction,
            outage_duration_ticks=outage_duration_ticks,
            verbose=False
        )
    
    # Get statistics
    stats = sim.get_stats()
    latency = sim.get_latency_statistics()
    
    # Compute derived metrics
    total_deliveries_exact = latency['total_deliveries'] if latency else 0
    total_bytes = stats['total_bytes']
    bytes_per_delivery = total_bytes / total_deliveries_exact if total_deliveries_exact > 0 else float('inf')
    
    result = {
        'D': D,
        'D_lazy': D_lazy,
        'churny_fraction': churny_fraction,
        'outage_region': outage_region,
        'outage_fraction': outage_fraction,
        'outage_duration_sec': outage_duration_sec,
        'seed': seed,
        # Primary objectives
        'delivery_rate': stats['delivery_mean'],
        'p99_latency_sec': latency['p99_latency_sec'] if latency else None,
        'bytes_per_delivery': bytes_per_delivery,
        # Secondary metrics
        'p50_latency_sec': latency['p50_latency_sec'] if latency else None,
        'p90_latency_sec': latency['p90_latency_sec'] if latency else None,
        'avg_latency_sec': latency['avg_latency_sec'] if latency else None,
        # Bandwidth breakdown
        'payload_bytes_MB': stats['payload_bytes'] / (1024 * 1024),
        'control_bytes_MB': stats['control_bytes'] / (1024 * 1024),
        'total_bytes_MB': stats['total_bytes'] / (1024 * 1024),
        'payload_pct': stats['payload_pct'],
        # Other stats
        'messages_published': stats['total_origin'],
        'total_deliveries': total_deliveries_exact,
        'drops_peer': stats['drops_peer'],
        'drops_offline': stats['drops_offline'],
        'online_rate': stats['online_total'] / 1000,
        'churn_events': stats['churn_events'],
    }
    
    return result


def run_multi_seed(config: dict, num_seeds: int = 3) -> dict:
    """
    Run configuration with multiple seeds and aggregate results.
    """
    results = []
    
    for seed in range(num_seeds):
        result = run_correlated_churn_experiment(
            D=config['D'],
            D_lazy=config['D_lazy'],
            churny_fraction=config['churny_fraction'],
            outage_region=config['outage_region'],
            outage_fraction=config['outage_fraction'],
            outage_duration_sec=config['outage_duration_sec'],
            seed=seed + 42
        )
        results.append(result)
    
    # Aggregate metrics
    metrics = [
        'delivery_rate', 'p99_latency_sec', 'bytes_per_delivery',
        'p50_latency_sec', 'p90_latency_sec', 'avg_latency_sec',
        'payload_bytes_MB', 'control_bytes_MB', 'total_bytes_MB', 'payload_pct',
        'messages_published', 'total_deliveries',
        'drops_peer', 'drops_offline', 'online_rate', 'churn_events'
    ]
    
    aggregated = {
        'D': config['D'],
        'D_lazy': config['D_lazy'],
        'churny_fraction': config['churny_fraction'],
        'outage_region': config['outage_region'],
        'outage_fraction': config['outage_fraction'],
        'outage_duration_sec': config['outage_duration_sec'],
        'num_seeds': num_seeds,
    }
    
    for metric in metrics:
        values = [r[metric] for r in results if r[metric] is not None]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
        else:
            aggregated[f'{metric}_mean'] = None
            aggregated[f'{metric}_std'] = None
    
    return aggregated


def run_config(config, num_seeds):
    """
    Worker function to run a single config and return the result.
    Must be at module level for ProcessPoolExecutor to pickle it.
    """
    try:
        result = run_multi_seed(config, num_seeds=num_seeds)
        return result
    except Exception as e:
        print(f"ERROR for D={config['D']}, D_lazy={config['D_lazy']}, "
              f"outage={config['outage_region']}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Correlated churn study for Lean Gossip paper')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of random seeds per config (default: 3)')
    parser.add_argument('--output', type=str, default='correlated_churn_results.csv',
                        help='Output CSV file')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced scenarios')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of parallel workers (default: 3)')
    
    args = parser.parse_args()
    
    # Configurations to compare (from paper)
    # GS(8,1), GS(8,3), GS(8,8) - Lean Gossip vs Ethereum
    configs_base = [
        {'D': 8, 'D_lazy': 1},   # Extreme Lean Gossip
        {'D': 8, 'D_lazy': 3},   # Recommended Lean Gossip
        {'D': 8, 'D_lazy': 8},   # Ethereum deployed
    ]
    
    # Outage scenarios
    if args.quick:
        outage_scenarios = [
            # (region, fraction, duration_sec)
            ('EU', 0.5, 10.0),   # 50% of EU offline for 10 seconds
        ]
        churny_fractions = [0.2]
    else:
        outage_scenarios = [
            # No outage baseline
            ('none', 0.0, 0.0),
            # Moderate regional outage: 30% of region for 10 seconds
            ('EU', 0.3, 10.0),
            ('US', 0.3, 10.0),
            ('Asia', 0.3, 10.0),
            # Severe regional outage: 50% of region for 10 seconds  
            ('EU', 0.5, 10.0),
            ('US', 0.5, 10.0),
            # Extended outage: 30% for 30 seconds
            ('EU', 0.3, 30.0),
        ]
        churny_fractions = [0.0, 0.2]  # No churn + 20% churn
    
    # Build full config list
    configs = []
    for base in configs_base:
        for churny_frac in churny_fractions:
            for region, frac, duration in outage_scenarios:
                config = {
                    'D': base['D'],
                    'D_lazy': base['D_lazy'],
                    'churny_fraction': churny_frac,
                    'outage_region': region,
                    'outage_fraction': frac,
                    'outage_duration_sec': duration,
                }
                configs.append(config)
    
    # Metrics for CSV
    metrics = [
        'delivery_rate', 'p99_latency_sec', 'bytes_per_delivery',
        'p50_latency_sec', 'p90_latency_sec', 'avg_latency_sec',
        'payload_bytes_MB', 'control_bytes_MB', 'total_bytes_MB', 'payload_pct',
        'messages_published', 'total_deliveries',
        'drops_peer', 'drops_offline', 'online_rate', 'churn_events'
    ]
    
    metric_fields = []
    for m in metrics:
        metric_fields.extend([f'{m}_mean', f'{m}_std'])
    
    fieldnames = ['D', 'D_lazy', 'churny_fraction', 'outage_region', 
                  'outage_fraction', 'outage_duration_sec', 'num_seeds'] + metric_fields

    # Check for existing results to resume
    output_file = args.output
    completed_configs = set()
    file_mode = 'w'

    if os.path.exists(output_file):
        file_mode = 'a'
        try:
            with open(output_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (int(row['D']), int(row['D_lazy']), 
                           float(row['churny_fraction']), row['outage_region'],
                           float(row['outage_fraction']), float(row['outage_duration_sec']))
                    completed_configs.add(key)
        except Exception as e:
            print(f"Warning: Could not read existing file {output_file}: {e}. Starting fresh.")
            file_mode = 'w'
            completed_configs = set()

    remaining_configs = len(configs) - len(completed_configs)
    num_workers = min(args.workers, remaining_configs) if remaining_configs > 0 else 1

    print("=" * 70)
    print("CORRELATED CHURN STUDY")
    print("=" * 70)
    print(f"Configurations: {len(configs_base)} × {len(churny_fractions)} churn levels × {len(outage_scenarios)} scenarios")
    print(f"Total: {len(configs)} configs × {args.seeds} seeds = {len(configs) * args.seeds} runs")
    print(f"Output: {args.output}")
    print(f"Workers: {num_workers}")
    if completed_configs:
        print(f"Resuming: {len(completed_configs)} configurations already completed")
    print("=" * 70)
    print()

    import concurrent.futures
    import threading
    
    lock = threading.Lock()
    completed_count = len(completed_configs)
    
    with open(output_file, file_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == 'w':
            writer.writeheader()
            f.flush()

        # Filter out already completed configs
        configs_to_run = []
        for config in configs:
            key = (config['D'], config['D_lazy'], config['churny_fraction'],
                   config['outage_region'], config['outage_fraction'], 
                   config['outage_duration_sec'])
            if key not in completed_configs:
                configs_to_run.append(config)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            futures = {executor.submit(run_config, config, args.seeds): config 
                       for config in configs_to_run}
            
            for future in concurrent.futures.as_completed(futures):
                config = futures[future]
                result = future.result()
                if result is not None:
                    with lock:
                        writer.writerow(result)
                        f.flush()
                        completed_count += 1
                    
                    print(f"[{completed_count}/{len(configs)}] D={config['D']}, D_lazy={config['D_lazy']}, "
                          f"churn={config['churny_fraction']:.0%}, "
                          f"outage={config['outage_region']} {config['outage_fraction']:.0%} for {config['outage_duration_sec']}s: "
                          f"delivery={result['delivery_rate_mean']:.1%}, "
                          f"p99={result['p99_latency_sec_mean']:.2f}s, "
                          f"cost={result['bytes_per_delivery_mean']/1536:.1f}×")
    
    print()
    print("=" * 70)
    print(f"Completed {completed_count}/{len(configs)} configurations")
    print(f"Saved: {args.output}")
    print("=" * 70)
    
    # Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: GS(8,3) vs GS(8,8) under correlated churn")
    print("=" * 70)
    
    # Read back results for summary
    try:
        import pandas as pd
        df = pd.read_csv(args.output)
        
        # Filter to 20% churn scenarios
        df_churn = df[df['churny_fraction'] == 0.2]
        
        if len(df_churn) > 0:
            print(f"\n{'Scenario':<40} | {'GS(8,3) Del':<12} | {'GS(8,8) Del':<12} | {'Winner':<10}")
            print("-" * 80)
            
            for (region, frac, dur), group in df_churn.groupby(['outage_region', 'outage_fraction', 'outage_duration_sec']):
                gs83 = group[group['D_lazy'] == 3]['delivery_rate_mean'].values
                gs88 = group[group['D_lazy'] == 8]['delivery_rate_mean'].values
                
                if len(gs83) > 0 and len(gs88) > 0:
                    gs83_del = gs83[0]
                    gs88_del = gs88[0]
                    winner = "GS(8,3)" if gs83_del >= gs88_del else "GS(8,8)"
                    
                    if region == 'none':
                        scenario = "Baseline (no outage)"
                    else:
                        scenario = f"{region} {frac:.0%} offline for {dur:.0f}s"
                    
                    print(f"{scenario:<40} | {gs83_del:>10.2%} | {gs88_del:>10.2%} | {winner:<10}")
    except Exception as e:
        print(f"Could not generate summary: {e}")


if __name__ == "__main__":
    main()