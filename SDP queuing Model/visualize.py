import matplotlib.pyplot as plt
import numpy as np

def plot_theory_vs_simulation(theoretical_results, simulation_results):
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extracting N values and corresponding results
    theo_N = sorted(list(theoretical_results.keys()))
    sim_N = sorted(list(simulation_results.keys()))
    
    # Colors for different queues
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'D']
    queue_names = ['Controller', 'AH1', 'AH2', 'AH3']
    
    # 1. Throughput Comparison
    theo_throughput = [theoretical_results[n]['throughput'] for n in theo_N]
    sim_throughput = [simulation_results[n]['throughput'] for n in sim_N]
    
    ax1.plot(theo_N, theo_throughput, '-', color='blue', label='Theoretical', linewidth=2)
    ax1.scatter(sim_N, sim_throughput, color='blue', label='Simulation',
               marker='o', s=100, facecolor='white', linewidth=2)
    ax1.set_title('System Throughput Comparison')
    ax1.set_xlabel('Number of Users (N)')
    ax1.set_ylabel('Throughput')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Response Times for each queue
    ax2.set_yscale('log')
    for i in range(4):  # For each queue
        theo_response = [theoretical_results[n]['waiting_times'][i] for n in theo_N]
        sim_response = [simulation_results[n]['waiting_times'][i] for n in sim_N]
        
        ax2.plot(theo_N, theo_response, '-', color=colors[i],
                label=f'Theoretical ({queue_names[i]})', linewidth=2)
        ax2.scatter(sim_N, sim_response, color=colors[i],
                   label=f'Simulation ({queue_names[i]})',
                   marker=markers[i], s=100, facecolor='white', linewidth=2)
    
    ax2.set_title('Response Time Comparison (Log Scale)')
    ax2.set_xlabel('Number of Users (N)')
    ax2.set_ylabel('Response Time (log scale)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Server Utilization for each queue
    for i in range(4):
        theo_util = [theoretical_results[n]['utilization'][i] for n in theo_N]
        sim_util = [simulation_results[n]['utilization'][i] for n in sim_N]
        
        ax3.plot(theo_N, theo_util, '-', color=colors[i],
                label=f'Theoretical ({queue_names[i]})', linewidth=2)
        ax3.scatter(sim_N, sim_util, color=colors[i],
                   label=f'Simulation ({queue_names[i]})',
                   marker=markers[i], s=100, facecolor='white', linewidth=2)
    
    ax3.set_title('Server Utilization Comparison')
    ax3.set_xlabel('Number of Users (N)')
    ax3.set_ylabel('Utilization')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Queue Length for each queue
    for i in range(4):
        theo_queue = [theoretical_results[n]['queue_lengths'][i] for n in theo_N]
        sim_queue = [simulation_results[n]['queue_lengths'][i] for n in sim_N]
        
        ax4.plot(theo_N, theo_queue, '-', color=colors[i],
                label=f'Theoretical ({queue_names[i]})', linewidth=2)
        ax4.scatter(sim_N, sim_queue, color=colors[i],
                   label=f'Simulation ({queue_names[i]})',
                   marker=markers[i], s=100, facecolor='white', linewidth=2)
    
    ax4.set_title('Queue Length Comparison')
    ax4.set_xlabel('Number of Users (N)')
    ax4.set_ylabel('Queue Length')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# plot_theory_vs_simulation(theoretical_results, simulation_results)
plot_theory_vs_simulation(theoretical_results, simulation_results)
