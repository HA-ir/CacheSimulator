import random
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import time

class CacheLevel:
    def __init__(self, level_id, cache_size, block_size, associativity, replacement_policy, access_time):
        """
        Initialize a cache level
        
        Args:
            level_id (int): The level of this cache (1 for L1, 2 for L2, etc.)
            cache_size (int): Total cache size in bytes
            block_size (int): Size of each cache block/line in bytes
            associativity (int): Associativity factor (1 for direct-mapped, -1 for fully associative)
            replacement_policy (str): 'LRU', 'FIFO', 'LFU', or 'RANDOM'
            access_time (int): Time (in cycles) to access this cache level
        """
        self.level_id = level_id
        self.cache_size = cache_size
        self.block_size = block_size
        self.associativity = associativity
        self.replacement_policy = replacement_policy.upper()
        self.access_time = access_time
        
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
        
        # Counters
        self.hits = 0
        self.misses = 0
        self.accesses = 0
        self.clock = 0  # Used for LRU and FIFO policies
        
        self.hit_addresses = []
        self.miss_addresses = []
        
        cache_type = "Direct-Mapped" if associativity == 1 else \
                    "Fully Associative" if associativity == -1 else \
                    f"{associativity}-Way Set Associative"
                    
        print(f"Initialized L{level_id} Cache:")
        print(f"  Cache Size: {cache_size} bytes")
        print(f"  Block Size: {block_size} bytes")
        print(f"  Type: {cache_type}")
        print(f"  Replacement Policy: {replacement_policy}")
        print(f"  Access Time: {access_time} cycles")
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
        
        return False
    
    def insert(self, address):
        """
        Insert a block into the cache after a miss
        
        Args:
            address (int): Address to insert
        """
        tag = self._get_tag(address)
        set_index = self._get_set_index(address)
        cache_set = self.cache[set_index]
        
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
        
        # Reset cache
        self.cache = [[] for _ in range(self.num_sets)]
    
    def print_stats(self):
        """Print cache statistics"""
        print(f"\nL{self.level_id} Cache Statistics:")
        print(f"  Total Accesses: {self.accesses}")
        print(f"  Hits: {self.hits}")
        print(f"  Misses: {self.misses}")
        print(f"  Hit Rate: {self.get_hit_rate():.4f}")
        print(f"  Miss Rate: {self.get_miss_rate():.4f}")


class MultiLevelCacheSimulator:
    def __init__(self, cache_levels, memory_access_time=100, inclusive=True):
        """
        Initialize a multi-level cache simulator
        
        Args:
            cache_levels (list): List of CacheLevel objects in order (L1, L2, L3, ...)
            memory_access_time (int): Time (in cycles) to access main memory
            inclusive (bool): Whether the cache hierarchy is inclusive or exclusive
        """
        self.cache_levels = cache_levels
        self.memory_access_time = memory_access_time
        self.inclusive = inclusive
        
        # Validate that block sizes are compatible across levels
        block_sizes = [cache.block_size for cache in cache_levels]
        for i in range(1, len(block_sizes)):
            if block_sizes[i] < block_sizes[i-1]:
                raise ValueError(f"Block size at L{i+1} ({block_sizes[i]}) cannot be smaller than L{i} ({block_sizes[i-1]})")
        
        # Statistics
        self.total_accesses = 0
        self.total_access_time = 0
        self.access_trace = []
        
        print("\nInitialized Multi-Level Cache Simulator:")
        print(f"  Number of Cache Levels: {len(cache_levels)}")
        print(f"  Memory Access Time: {memory_access_time} cycles")
        print(f"  Inclusion Policy: {'Inclusive' if inclusive else 'Exclusive'}")
    
    def access(self, address):
        """
        Access memory address through the cache hierarchy
        
        Args:
            address (int): Memory address to access
            
        Returns:
            tuple: (hit_level, access_time) where hit_level is the level where the data was found
                  (0 for memory) and access_time is the time taken for this access
        """
        self.total_accesses += 1
        self.access_trace.append(address)
        
        access_time = 0
        hit_level = 0  # 0 means main memory
        
        # Try to find the data in each cache level
        for level_idx, cache in enumerate(self.cache_levels):
            level_num = level_idx + 1  # Level number (1-based)
            
            # Check if this level has the data
            if cache.access(address):
                # Hit at this level
                hit_level = level_num
                access_time += cache.access_time
                break
            else:
                # Miss at this level, but count the access time
                access_time += cache.access_time
        
        # If not found in any cache level, access main memory
        if hit_level == 0:
            access_time += self.memory_access_time
        
        # Now update the cache hierarchy (insert the block in all levels up to where it was found)
        if hit_level != 1:  # If it wasn't in L1, we need to insert it there
            # For inclusive caches, insert in all levels
            if self.inclusive:
                for level_idx in range(hit_level - 1 if hit_level > 0 else len(self.cache_levels)):
                    self.cache_levels[level_idx].insert(address)
            else:
                # For exclusive caches, only insert in L1
                self.cache_levels[0].insert(address)
        
        self.total_access_time += access_time
        return hit_level, access_time
    
    def get_average_access_time(self):
        """Calculate the average memory access time"""
        if self.total_accesses == 0:
            return 0
        return self.total_access_time / self.total_accesses
    
    def print_stats(self):
        """Print statistics for all cache levels"""
        print("\nMulti-Level Cache Statistics:")
        print(f"  Total Memory Accesses: {self.total_accesses}")
        print(f"  Total Access Time: {self.total_access_time} cycles")
        print(f"  Average Access Time: {self.get_average_access_time():.2f} cycles")
        
        for cache in self.cache_levels:
            cache.print_stats()
    
    def reset_stats(self):
        """Reset statistics for all cache levels"""
        self.total_accesses = 0
        self.total_access_time = 0
        self.access_trace = []
        
        for cache in self.cache_levels:
            cache.reset_stats()
    
    def plot_hit_rates(self):
        """Plot hit rates for all cache levels"""
        plt.figure(figsize=(10, 6))
        
        labels = [f"L{i+1}" for i in range(len(self.cache_levels))]
        hit_rates = [cache.get_hit_rate() for cache in self.cache_levels]
        miss_rates = [cache.get_miss_rate() for cache in self.cache_levels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, hit_rates, width, label='Hit Rate')
        plt.bar(x + width/2, miss_rates, width, label='Miss Rate')
        
        plt.xlabel('Cache Level')
        plt.ylabel('Rate')
        plt.title('Hit and Miss Rates by Cache Level')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_access_distribution(self):
        """Plot access distribution across cache levels"""
        plt.figure(figsize=(10, 6))
        
        # Count accesses at each level
        level_accesses = [0] * (len(self.cache_levels) + 1)  # +1 for main memory
        level_labels = [f"L{i+1}" for i in range(len(self.cache_levels))] + ["Memory"]
        
        for cache in self.cache_levels:
            level_accesses[cache.level_id - 1] = cache.hits
        
        # Memory accesses = misses from last level cache
        if self.cache_levels:
            level_accesses[-1] = self.cache_levels[-1].misses
        
        plt.bar(level_labels, level_accesses)
        plt.xlabel('Memory Hierarchy Level')
        plt.ylabel('Number of Accesses')
        plt.title('Access Distribution Across Memory Hierarchy')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()


def generate_memory_trace(pattern_type, num_accesses, address_range):
    """
    Generate a memory access trace based on a specific pattern
    
    Args:
        pattern_type (str): 'sequential', 'random', 'loop', 'locality', or 'mixed'
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
    
    elif pattern_type == 'mixed':
        # Mix of different patterns
        patterns = ['sequential', 'random', 'loop', 'locality']
        segment_size = num_accesses // len(patterns)
        
        for pattern in patterns:
            segment_trace = generate_memory_trace(pattern, segment_size, address_range)
            trace.extend(segment_trace)
        
        # Add remaining accesses as random
        remaining = num_accesses - len(trace)
        if remaining > 0:
            random_trace = generate_memory_trace('random', remaining, address_range)
            trace.extend(random_trace)
    
    return trace


def analyze_multilevel_cache(trace, l1_size, l2_size, l3_size, block_sizes, associativities, policies, 
                         l1_time=1, l2_time=5, l3_time=10, mem_time=100, inclusive=True):
    """
    Analyze the performance of a multi-level cache configuration
    
    Args:
        trace (list): Memory access trace
        l1_size (int): L1 cache size in bytes
        l2_size (int): L2 cache size in bytes
        l3_size (int): L3 cache size in bytes
        block_sizes (list): Block sizes for each level [l1_block, l2_block, l3_block]
        associativities (list): Associativities for each level [l1_assoc, l2_assoc, l3_assoc]
        policies (list): Replacement policies for each level [l1_policy, l2_policy, l3_policy]
        l1_time (int): L1 access time in cycles
        l2_time (int): L2 access time in cycles
        l3_time (int): L3 access time in cycles
        mem_time (int): Main memory access time in cycles
        inclusive (bool): Whether the cache hierarchy is inclusive
        
    Returns:
        tuple: (hit_rates, avg_access_time) for the configuration
    """
    # Create cache levels
    l1_cache = CacheLevel(1, l1_size, block_sizes[0], associativities[0], policies[0], l1_time)
    l2_cache = CacheLevel(2, l2_size, block_sizes[1], associativities[1], policies[1], l2_time)
    l3_cache = CacheLevel(3, l3_size, block_sizes[2], associativities[2], policies[2], l3_time)
    
    # Create multi-level cache simulator
    simulator = MultiLevelCacheSimulator([l1_cache, l2_cache, l3_cache], mem_time, inclusive)
    
    # Run simulation
    for addr in trace:
        simulator.access(addr)
    
    # Collect results
    hit_rates = [cache.get_hit_rate() for cache in simulator.cache_levels]
    avg_access_time = simulator.get_average_access_time()
    
    return hit_rates, avg_access_time


def compare_inclusion_policies(trace, l1_size, l2_size, l3_size, block_size, associativity, policy):
    """
    Compare inclusive vs exclusive cache hierarchies
    
    Args:
        trace (list): Memory access trace
        l1_size, l2_size, l3_size (int): Cache sizes in bytes
        block_size (int): Block size in bytes
        associativity (int): Associativity
        policy (str): Replacement policy
        
    Returns:
        dict: Results for inclusive and exclusive policies
    """
    results = {}
    
    # Test inclusive hierarchy
    l1_cache = CacheLevel(1, l1_size, block_size, associativity, policy, 1)
    l2_cache = CacheLevel(2, l2_size, block_size, associativity, policy, 5)
    l3_cache = CacheLevel(3, l3_size, block_size, associativity, policy, 10)
    
    inclusive_simulator = MultiLevelCacheSimulator([l1_cache, l2_cache, l3_cache], 100, True)
    
    # Run simulation
    for addr in trace:
        inclusive_simulator.access(addr)
    
    # Store results
    results['Inclusive'] = {
        'l1_hit_rate': l1_cache.get_hit_rate(),
        'l2_hit_rate': l2_cache.get_hit_rate(),
        'l3_hit_rate': l3_cache.get_hit_rate(),
        'avg_access_time': inclusive_simulator.get_average_access_time()
    }
    
    # Reset for exclusive hierarchy
    l1_cache = CacheLevel(1, l1_size, block_size, associativity, policy, 1)
    l2_cache = CacheLevel(2, l2_size, block_size, associativity, policy, 5)
    l3_cache = CacheLevel(3, l3_size, block_size, associativity, policy, 10)
    
    exclusive_simulator = MultiLevelCacheSimulator([l1_cache, l2_cache, l3_cache], 100, False)
    
    # Run simulation
    for addr in trace:
        exclusive_simulator.access(addr)
    
    # Store results
    results['Exclusive'] = {
        'l1_hit_rate': l1_cache.get_hit_rate(),
        'l2_hit_rate': l2_cache.get_hit_rate(),
        'l3_hit_rate': l3_cache.get_hit_rate(),
        'avg_access_time': exclusive_simulator.get_average_access_time()
    }
    
    return results


def compare_hierarchies(trace):
    """
    Compare different cache hierarchy configurations
    
    Args:
        trace (list): Memory access trace
        
    Returns:
        dict: Results for different configurations
    """
    configs = [
        # L1 only
        {'name': 'L1 Only', 'levels': 1, 'sizes': [32768, 0, 0]},
        # L1 + L2
        {'name': 'L1 + L2', 'levels': 2, 'sizes': [8192, 65536, 0]},
        # L1 + L2 + L3
        {'name': 'L1 + L2 + L3', 'levels': 3, 'sizes': [8192, 65536, 524288]},
        # Small L1, Large L2
        {'name': 'Small L1, Large L2', 'levels': 2, 'sizes': [4096, 131072, 0]},
        # Large L1, Small L2
        {'name': 'Large L1, Small L2', 'levels': 2, 'sizes': [16384, 32768, 0]},
    ]
    
    results = {}
    
    # Common parameters
    block_size = 64
    associativity = 4
    policy = 'LRU'
    
    for config in configs:
        # Create cache levels based on configuration
        cache_levels = []
        for i in range(config['levels']):
            if config['sizes'][i] > 0:
                access_time = 1 if i == 0 else 5 if i == 1 else 10
                cache_levels.append(CacheLevel(i+1, config['sizes'][i], block_size, associativity, policy, access_time))
        
        # Create simulator
        simulator = MultiLevelCacheSimulator(cache_levels, 100, True)
        
        # Run simulation
        for addr in trace:
            simulator.access(addr)
        
        # Store results
        results[config['name']] = {
            'avg_access_time': simulator.get_average_access_time(),
            'hit_rates': [cache.get_hit_rate() for cache in cache_levels]
        }
    
    return results


def run_multilevel_analysis():
    """Run a comprehensive analysis of multi-level cache configurations"""
    # Define parameters
    access_patterns = ['sequential', 'random', 'loop', 'locality', 'mixed']
    num_accesses = 100000
    address_range = 1048576  # 1MB address space
    
    # Generate traces for each access pattern
    traces = {}
    for pattern in access_patterns:
        traces[pattern] = generate_memory_trace(pattern, num_accesses, address_range)
    
    # Compare cache hierarchies for each pattern
    hierarchy_results = {}
    for pattern in access_patterns:
        print(f"\nAnalyzing cache hierarchies for {pattern} access pattern...")
        hierarchy_results[pattern] = compare_hierarchies(traces[pattern])
    
    # Print summary
    print("\nAverage Access Time (cycles) by Configuration and Access Pattern:")
    print("Configuration".ljust(20), end="")
    for pattern in access_patterns:
        print(f"{pattern.capitalize().ljust(12)}", end="")
    print()
    
    for config in list(hierarchy_results[access_patterns[0]].keys()):
        print(config.ljust(20), end="")
        for pattern in access_patterns:
            avg_time = hierarchy_results[pattern][config]['avg_access_time']
            print(f"{avg_time:.2f}".ljust(12), end="")
        print()
    
    # Compare inclusion policies
    print("\nComparing Inclusion Policies...")
    policy_results = {}
    for pattern in access_patterns:
        policy_results[pattern] = compare_inclusion_policies(
            traces[pattern], 8192, 65536, 524288, 64, 4, 'LRU'
        )
    
    # Print inclusion policy summary
    print("\nAverage Access Time (cycles) by Inclusion Policy and Access Pattern:")
    print("Policy".ljust(12), end="")
    for pattern in access_patterns:
        print(f"{pattern.capitalize().ljust(12)}", end="")
    print()
    
    for policy in ['Inclusive', 'Exclusive']:
        print(policy.ljust(12), end="")
        for pattern in access_patterns:
            avg_time = policy_results[pattern][policy]['avg_access_time']
            print(f"{avg_time:.2f}".ljust(12), end="")
        print()
    
    # Return results for further analysis if needed
    return hierarchy_results, policy_results, traces


def main():
    parser = argparse.ArgumentParser(description='Multi-Level Cache Simulator')
    parser.add_argument('--pattern', type=str, default='mixed',
                        choices=['sequential', 'random', 'loop', 'locality', 'mixed'],
                        help='Memory access pattern')
    parser.add_argument('--num-accesses', type=int, default=100000, help='Number of memory accesses to simulate')
    parser.add_argument('--address-range', type=int, default=1048576, help='Range of addresses to access')
    parser.add_argument('--l1-size', type=int, default=32768, help='L1 cache size in bytes')
    parser.add_argument('--l2-size', type=int, default=262144, help='L2 cache size in bytes')
    parser.add_argument('--l3-size', type=int, default=4194304, help='L3 cache size in bytes')
    parser.add_argument('--block-size', type=int, default=64, help='Block size in bytes')
    parser.add_argument('--associativity', type=int, default=4, help='Cache associativity')
    parser.add_argument('--policy', type=str, default='LRU', choices=['LRU', 'FIFO', 'LFU', 'RANDOM'],
                        help='Replacement policy')
    parser.add_argument('--inclusive', action='store_true', help='Use inclusive cache hierarchy')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'compare-hierarchies', 'compare-policies', 'comprehensive'],
                        help='Simulation mode')
    
    args = parser.parse_args()
    
    if args.mode == 'comprehensive':
        run_multilevel_analysis()
        return
    
    # Generate memory access trace
    trace = generate_memory_trace(args.pattern, args.num_accesses, args.address_range)
    
    if args.mode == 'single':
        # Create cache levels
        l1_cache = CacheLevel(1, args.l1_size, args.block_size, args.associativity, args.policy, 1)
        l2_cache = CacheLevel(2, args.l2_size, args.block_size, args.associativity, args.policy, 5)
        l3_cache = CacheLevel(3, args.l3_size, args.block_size, args.associativity, args.policy, 10)
        
        # Create multi-level cache simulator
        simulator = MultiLevelCacheSimulator([l1_cache, l2_cache, l3_cache], 100, args.inclusive)
        
        # Run simulation
        start_time = time.time()
        for addr in trace:
            simulator.access(addr)
        end_time = time.time()
        
        # Print statistics
        simulator.print_stats()
        print(f"\nSimulation Time: {end_time - start_time:.2f} seconds")
        
        # Plot results
        simulator.plot_hit_rates()
        simulator.plot_access_distribution()
    
    elif args.mode == 'compare-hierarchies':
        # Compare different cache hierarchies
        results = compare_hierarchies(trace)
        
        # Print results
        print("\nComparison of Cache Hierarchies:")
        for config, data in results.items():
            hit_rates_str = ", ".join([f"L{i+1}: {rate:.4f}" for i, rate in enumerate(data['hit_rates'])])
            print(f"  {config}:")
            print(f"    Hit Rates: {hit_rates_str}")
            print(f"    Average Access Time: {data['avg_access_time']:.2f} cycles")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        configs = list(results.keys())
        avg_times = [results[config]['avg_access_time'] for config in configs]
        
        plt.bar(configs, avg_times)
        plt.xlabel('Cache Hierarchy Configuration')
        plt.ylabel('Average Access Time (cycles)')
        plt.title('Average Access Time by Cache Hierarchy')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plt.show()
    
    elif args.mode == 'compare-policies':
        # Compare inclusive vs exclusive policies
        results = compare_inclusion_policies(trace, args.l1_size, args.l2_size, args.l3_size, 
                                         args.block_size, args.associativity, args.policy)
        
        # Print results
        print("\nComparison of Inclusion Policies:")
        for policy, data in results.items():
            print(f"  {policy} Policy:")
            print(f"    L1 Hit Rate: {data['l1_hit_rate']:.4f}")
            print(f"    L2 Hit Rate: {data['l2_hit_rate']:.4f}")
            print(f"    L3 Hit Rate: {data['l3_hit_rate']:.4f}")
            print(f"    Average Access Time: {data['avg_access_time']:.2f} cycles")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        policies = list(results.keys())
        l1_rates = [results[policy]['l1_hit_rate'] for policy in policies]
        l2_rates = [results[policy]['l2_hit_rate'] for policy in policies]
        l3_rates = [results[policy]['l3_hit_rate'] for policy in policies]
        avg_times = [results[policy]['avg_access_time'] for policy in policies]
        
        x = np.arange(len(policies))
        width = 0.2
        
        plt.bar(x - width, l1_rates, width, label='L1 Hit Rate')
        plt.bar(x, l2_rates, width, label='L2 Hit Rate')
        plt.bar(x + width, l3_rates, width, label='L3 Hit Rate')
        
        plt.xlabel('Inclusion Policy')
        plt.ylabel('Hit Rate')
        plt.title('Cache Hit Rates by Inclusion Policy')
        plt.xticks(x, policies)
        plt.legend()
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Plot average access time
        plt.figure(figsize=(10, 6))
        plt.bar(policies, avg_times)
        plt.xlabel('Inclusion Policy')
        plt.ylabel('Average Access Time (cycles)')
        plt.title('Average Access Time by Inclusion Policy')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()