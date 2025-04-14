import random
import argparse
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np

class CacheSimulator:
    def __init__(self, cache_size, block_size, associativity, replacement_policy):
        """
        Initialize the cache simulator with given parameters
        
        Args:
            cache_size (int): Total cache size in bytes
            block_size (int): Size of each cache block/line in bytes
            associativity (int): Associativity factor (1 for direct-mapped, -1 for fully associative)
            replacement_policy (str): 'LRU', 'FIFO', 'LFU', or 'RANDOM'
        """
        self.cache_size = cache_size
        self.block_size = block_size
        self.associativity = associativity
        self.replacement_policy = replacement_policy.upper()
        
        # Calculate cache organization parameters
        self.num_blocks = cache_size // block_size
        
        if associativity == -1:  # Fully associative
            self.num_sets = 1
            self.set_size = self.num_blocks
        else:
            self.set_size = associativity
            self.num_sets = self.num_blocks // self.set_size
        
        # Initialize cache structure
        self.cache = [[] for _ in range(self.num_sets)]
        
        # For each cache line, we'll store a tuple (tag, metadata)
        # metadata depends on replacement policy:
        # - LRU: counter for last access
        # - FIFO: counter for insertion time
        # - LFU: counter for access frequency
        # - RANDOM: no additional metadata needed
        
        # Counters
        self.hits = 0
        self.misses = 0
        self.accesses = 0
        self.clock = 0  # Used for LRU and FIFO policies
        
        # Statistics tracking
        self.hit_addresses = []
        self.miss_addresses = []
        self.access_trace = []
        
        print(f"Initialized Cache Simulator:")
        print(f"  Cache Size: {cache_size} bytes")
        print(f"  Block Size: {block_size} bytes")
        if associativity == 1:
            print(f"  Type: Direct-Mapped")
        elif associativity == -1:
            print(f"  Type: Fully Associative")
        else:
            print(f"  Type: {associativity}-Way Set Associative")
        print(f"  Replacement Policy: {replacement_policy}")
        print(f"  Number of Sets: {self.num_sets}")
        print(f"  Number of Blocks: {self.num_blocks}")

    def _get_set_index(self, address):
        """Calculate the set index for an address"""
        # Extract block number (remove offset bits)
        block_number = address // self.block_size
        
        if self.associativity == -1:  # Fully associative
            return 0
        else:
            # Extract set index bits
            return block_number % self.num_sets
    
    def _get_tag(self, address):
        """Calculate the tag for an address"""
        # Extract block number (remove offset bits)
        block_number = address // self.block_size
        
        if self.associativity == -1:  # Fully associative
            return block_number
        else:
            # Extract tag bits
            return block_number // self.num_sets
    
    def access(self, address):
        """
        Simulate a memory access to the given address
        
        Args:
            address (int): Memory address to access
            
        Returns:
            bool: True if hit, False if miss
        """
        self.accesses += 1
        self.clock += 1
        self.access_trace.append(address)
        
        tag = self._get_tag(address)
        set_index = self._get_set_index(address)
        cache_set = self.cache[set_index]
        
        # Check if we have a hit
        for i, (block_tag, metadata) in enumerate(cache_set):
            if block_tag == tag:
                # Hit: Update metadata based on replacement policy
                if self.replacement_policy == 'LRU':
                    cache_set[i] = (tag, self.clock)
                elif self.replacement_policy == 'LFU':
                    cache_set[i] = (tag, metadata + 1)
                # FIFO: No update needed on hit
                
                self.hits += 1
                self.hit_addresses.append(address)
                return True
        
        # Miss: Need to insert the block
        self.misses += 1
        self.miss_addresses.append(address)
        
        # Initialize metadata based on replacement policy
        if self.replacement_policy == 'LRU':
            metadata = self.clock
        elif self.replacement_policy == 'FIFO':
            metadata = self.clock
        elif self.replacement_policy == 'LFU':
            metadata = 1
        else:  # RANDOM
            metadata = None
        
        # If set isn't full, simply add the block
        if len(cache_set) < self.set_size:
            cache_set.append((tag, metadata))
        else:
            # Need to replace a block based on the policy
            if self.replacement_policy == 'LRU':
                # Replace the least recently used
                lru_index = min(range(len(cache_set)), key=lambda i: cache_set[i][1])
                cache_set[lru_index] = (tag, metadata)
            elif self.replacement_policy == 'FIFO':
                # Replace the first inserted
                fifo_index = min(range(len(cache_set)), key=lambda i: cache_set[i][1])
                cache_set[fifo_index] = (tag, metadata)
            elif self.replacement_policy == 'LFU':
                # Replace the least frequently used
                lfu_index = min(range(len(cache_set)), key=lambda i: cache_set[i][1])
                cache_set[lfu_index] = (tag, metadata)
            else:  # RANDOM
                # Replace a random entry
                random_index = random.randint(0, len(cache_set) - 1)
                cache_set[random_index] = (tag, metadata)
        
        return False
    
    def get_hit_rate(self):
        """Calculate the cache hit rate"""
        if self.accesses == 0:
            return 0
        return self.hits / self.accesses
    
    def get_miss_rate(self):
        """Calculate the cache miss rate"""
        if self.accesses == 0:
            return 0
        return self.misses / self.accesses
    
    def reset_stats(self):
        """Reset hit/miss statistics"""
        self.hits = 0
        self.misses = 0
        self.accesses = 0
        self.clock = 0
        self.hit_addresses = []
        self.miss_addresses = []
        self.access_trace = []
        
        # Reset cache
        self.cache = [[] for _ in range(self.num_sets)]
    
    def print_stats(self):
        """Print cache statistics"""
        print("\nCache Statistics:")
        print(f"  Total Accesses: {self.accesses}")
        print(f"  Hits: {self.hits}")
        print(f"  Misses: {self.misses}")
        print(f"  Hit Rate: {self.get_hit_rate():.4f}")
        print(f"  Miss Rate: {self.get_miss_rate():.4f}")
    
    def plot_access_pattern(self, output_file=None):
        """
        Plot the access pattern with hits and misses
        
        Args:
            output_file (str, optional): File to save the plot to
        """
        plt.figure(figsize=(12, 6))
        
        # Plot all accesses
        x = list(range(len(self.access_trace)))
        plt.scatter(x, self.access_trace, c='gray', s=10, alpha=0.3, label='All Accesses')
        
        # Plot hits
        if self.hit_addresses:
            hit_indices = []
            hit_addresses = []
            for i, addr in enumerate(self.access_trace):
                if addr in self.hit_addresses and self.hit_addresses.count(addr) > 0:
                    hit_indices.append(i)
                    hit_addresses.append(addr)
                    self.hit_addresses.remove(addr)  # Remove to handle duplicate addresses
            plt.scatter(hit_indices, hit_addresses, c='green', s=20, label='Cache Hits')
        
        # Plot misses
        if self.miss_addresses:
            miss_indices = []
            miss_addresses = []
            for i, addr in enumerate(self.access_trace):
                if addr in self.miss_addresses and self.miss_addresses.count(addr) > 0:
                    miss_indices.append(i)
                    miss_addresses.append(addr)
                    self.miss_addresses.remove(addr)  # Remove to handle duplicate addresses
            plt.scatter(miss_indices, miss_addresses, c='red', s=20, label='Cache Misses')
        
        plt.title('Memory Access Pattern with Cache Hits and Misses')
        plt.xlabel('Access Number')
        plt.ylabel('Memory Address')
        plt.legend()
        plt.grid(True)
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()


def generate_memory_trace(pattern_type, num_accesses, address_range):
    """
    Generate a memory access trace based on a specific pattern
    
    Args:
        pattern_type (str): 'sequential', 'random', 'loop', or 'locality'
        num_accesses (int): Number of memory accesses to generate
        address_range (int): Range of addresses (0 to address_range-1)
        
    Returns:
        list: List of memory addresses to access
    """
    trace = []
    
    if pattern_type == 'sequential':
        # Sequential access pattern
        for i in range(num_accesses):
            trace.append(i % address_range)
    
    elif pattern_type == 'random':
        # Random access pattern
        for _ in range(num_accesses):
            trace.append(random.randint(0, address_range - 1))
    
    elif pattern_type == 'loop':
        # Loop access pattern (repeatedly access a fixed range)
        loop_size = min(100, address_range)
        loop_start = random.randint(0, address_range - loop_size)
        loop_addresses = list(range(loop_start, loop_start + loop_size))
        
        for i in range(num_accesses):
            idx = i % loop_size
            trace.append(loop_addresses[idx])
    
    elif pattern_type == 'locality':
        # Temporal and spatial locality
        # 80% of accesses in 20% of address space
        hot_region_size = int(0.2 * address_range)
        hot_region_start = random.randint(0, address_range - hot_region_size)
        
        for _ in range(num_accesses):
            if random.random() < 0.8:  # 80% probability of hot region
                addr = random.randint(hot_region_start, hot_region_start + hot_region_size - 1)
            else:
                # Access outside hot region
                if random.random() < 0.5 and hot_region_start > 0:
                    # Before hot region
                    addr = random.randint(0, hot_region_start - 1)
                else:
                    # After hot region
                    addr = random.randint(hot_region_start + hot_region_size, address_range - 1)
            trace.append(addr)
    
    return trace


def compare_policies(trace, cache_size, block_size, associativity):
    """
    Compare different replacement policies for the same trace and cache configuration
    
    Args:
        trace (list): Memory access trace
        cache_size (int): Cache size in bytes
        block_size (int): Block size in bytes
        associativity (int): Associativity factor
        
    Returns:
        dict: Dictionary of hit rates for each policy
    """
    policies = ['LRU', 'FIFO', 'LFU', 'RANDOM']
    results = {}
    
    for policy in policies:
        simulator = CacheSimulator(cache_size, block_size, associativity, policy)
        for addr in trace:
            simulator.access(addr)
        
        results[policy] = {
            'hit_rate': simulator.get_hit_rate(),
            'miss_rate': simulator.get_miss_rate(),
            'hits': simulator.hits,
            'misses': simulator.misses
        }
        
        print(f"\n{policy} Policy Results:")
        simulator.print_stats()
    
    return results


def compare_associativities(trace, cache_size, block_size, policy):
    """
    Compare different associativity levels for the same trace, cache size, and policy
    
    Args:
        trace (list): Memory access trace
        cache_size (int): Cache size in bytes
        block_size (int): Block size in bytes
        policy (str): Replacement policy
        
    Returns:
        dict: Dictionary of hit rates for each associativity level
    """
    # Test direct-mapped, 2-way, 4-way, 8-way, and fully associative
    associativities = [1, 2, 4, 8, -1]  # -1 for fully associative
    results = {}
    
    for assoc in associativities:
        if assoc == -1:
            assoc_name = "Fully Associative"
        elif assoc == 1:
            assoc_name = "Direct-Mapped"
        else:
            assoc_name = f"{assoc}-Way Set Associative"
            
        print(f"\nTesting {assoc_name}:")
        simulator = CacheSimulator(cache_size, block_size, assoc, policy)
        for addr in trace:
            simulator.access(addr)
        
        results[assoc_name] = {
            'hit_rate': simulator.get_hit_rate(),
            'miss_rate': simulator.get_miss_rate(),
            'hits': simulator.hits,
            'misses': simulator.misses
        }
        
        simulator.print_stats()
    
    return results


def compare_block_sizes(trace, cache_size, associativity, policy):
    """
    Compare different block sizes for the same trace, cache size, and associativity
    
    Args:
        trace (list): Memory access trace
        cache_size (int): Cache size in bytes
        associativity (int): Associativity factor
        policy (str): Replacement policy
        
    Returns:
        dict: Dictionary of hit rates for each block size
    """
    # Test different block sizes (in bytes)
    block_sizes = [16, 32, 64, 128, 256]
    results = {}
    
    for block_size in block_sizes:
        print(f"\nTesting Block Size {block_size} bytes:")
        simulator = CacheSimulator(cache_size, block_size, associativity, policy)
        for addr in trace:
            simulator.access(addr)
        
        results[block_size] = {
            'hit_rate': simulator.get_hit_rate(),
            'miss_rate': simulator.get_miss_rate(),
            'hits': simulator.hits,
            'misses': simulator.misses
        }
        
        simulator.print_stats()
    
    return results


def plot_comparison(results, title, xlabel, ylabel='Hit Rate'):
    """
    Plot comparison of results
    
    Args:
        results (dict): Dictionary of results
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    plt.figure(figsize=(10, 6))
    
    categories = list(results.keys())
    hit_rates = [results[cat]['hit_rate'] for cat in categories]
    miss_rates = [results[cat]['miss_rate'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, hit_rates, width, label='Hit Rate')
    plt.bar(x + width/2, miss_rates, width, label='Miss Rate')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, categories, rotation=45 if len(categories[0]) > 5 else 0)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.show()


def run_comprehensive_analysis():
    """Run a comprehensive analysis with various cache configurations and access patterns"""
    # Define parameters
    cache_sizes = [1024, 4096, 16384]  # 1KB, 4KB, 16KB
    block_sizes = [32, 64, 128]
    associativities = [1, 2, 4, -1]  # 1 = direct-mapped, -1 = fully associative
    policies = ['LRU', 'FIFO', 'LFU', 'RANDOM']
    access_patterns = ['sequential', 'random', 'loop', 'locality']
    
    # Define number of addresses to access
    num_accesses = 10000
    address_range = 32768  # 32KB address space
    
    # Store results
    all_results = {}
    
    # Generate traces for each access pattern
    traces = {}
    for pattern in access_patterns:
        traces[pattern] = generate_memory_trace(pattern, num_accesses, address_range)
    
    # Run simulations
    print("Running Comprehensive Cache Analysis...")
    for cache_size in cache_sizes:
        all_results[cache_size] = {}
        for block_size in block_sizes:
            all_results[cache_size][block_size] = {}
            for assoc in associativities:
                all_results[cache_size][block_size][assoc] = {}
                for policy in policies:
                    all_results[cache_size][block_size][assoc][policy] = {}
                    for pattern in access_patterns:
                        # Create simulator
                        simulator = CacheSimulator(cache_size, block_size, assoc, policy)
                        
                        # Run simulation
                        for addr in traces[pattern]:
                            simulator.access(addr)
                        
                        # Store results
                        all_results[cache_size][block_size][assoc][policy][pattern] = {
                            'hit_rate': simulator.get_hit_rate(),
                            'miss_rate': simulator.get_miss_rate(),
                            'hits': simulator.hits,
                            'misses': simulator.misses
                        }
    
    # Print summary of best configurations for each access pattern
    print("\nBest Cache Configurations for Each Access Pattern:")
    for pattern in access_patterns:
        best_hit_rate = 0
        best_config = None
        
        for cache_size in cache_sizes:
            for block_size in block_sizes:
                for assoc in associativities:
                    for policy in policies:
                        result = all_results[cache_size][block_size][assoc][policy][pattern]
                        if result['hit_rate'] > best_hit_rate:
                            best_hit_rate = result['hit_rate']
                            best_config = (cache_size, block_size, assoc, policy)
        
        # Print best configuration
        cache_size, block_size, assoc, policy = best_config
        if assoc == -1:
            assoc_str = "Fully Associative"
        elif assoc == 1:
            assoc_str = "Direct-Mapped"
        else:
            assoc_str = f"{assoc}-Way Set Associative"
        
        print(f"\nBest Configuration for {pattern.capitalize()} Access Pattern:")
        print(f"  Cache Size: {cache_size} bytes")
        print(f"  Block Size: {block_size} bytes")
        print(f"  Type: {assoc_str}")
        print(f"  Replacement Policy: {policy}")
        print(f"  Hit Rate: {best_hit_rate:.4f}")
    
    # Return all results for further analysis if needed
    return all_results, traces


def main():
    parser = argparse.ArgumentParser(description='Cache Simulator')
    parser.add_argument('--cache-size', type=int, default=8192, help='Cache size in bytes')
    parser.add_argument('--block-size', type=int, default=64, help='Block size in bytes')
    parser.add_argument('--associativity', type=int, default=2,
                        help='Associativity (1 for direct-mapped, -1 for fully associative)')
    parser.add_argument('--policy', type=str, default='LRU', choices=['LRU', 'FIFO', 'LFU', 'RANDOM'],
                        help='Replacement policy')
    parser.add_argument('--num-accesses', type=int, default=10000, help='Number of memory accesses to simulate')
    parser.add_argument('--address-range', type=int, default=32768, help='Range of addresses to access')
    parser.add_argument('--pattern', type=str, default='random',
                        choices=['sequential', 'random', 'loop', 'locality'],
                        help='Memory access pattern')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'compare-policies', 'compare-associativities',
                                 'compare-block-sizes', 'comprehensive'],
                        help='Simulation mode')
    
    args = parser.parse_args()
    
    if args.mode == 'comprehensive':
        run_comprehensive_analysis()
        return
    
    # Generate memory access trace
    trace = generate_memory_trace(args.pattern, args.num_accesses, args.address_range)
    
    if args.mode == 'single':
        # Run a single simulation
        simulator = CacheSimulator(
            args.cache_size,
            args.block_size,
            args.associativity,
            args.policy
        )
        
        for addr in trace:
            simulator.access(addr)
        
        simulator.print_stats()
        simulator.plot_access_pattern()
    
    elif args.mode == 'compare-policies':
        # Compare different replacement policies
        results = compare_policies(trace, args.cache_size, args.block_size, args.associativity)
        plot_comparison(results, 'Comparison of Replacement Policies', 'Replacement Policy')
    
    elif args.mode == 'compare-associativities':
        # Compare different associativity levels
        results = compare_associativities(trace, args.cache_size, args.block_size, args.policy)
        plot_comparison(results, 'Comparison of Associativity Levels', 'Associativity')
    
    elif args.mode == 'compare-block-sizes':
        # Compare different block sizes
        results = compare_block_sizes(trace, args.cache_size, args.associativity, args.policy)
        plot_comparison(results, 'Comparison of Block Sizes', 'Block Size (bytes)')


if __name__ == '__main__':
    main()
