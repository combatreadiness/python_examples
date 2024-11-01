import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simpy
from collections import deque
import random

class JacksonNetworkMM1K:
    def __init__(self, lambda_0, lambda_1, lambda_2, lambda_3, mu_1, mu_2, mu_3, mu_4, 
                 transition_matrix, K1, K2, K3, K4, N=1):
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2 
        self.lambda_3 = lambda_3
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3
        self.mu_4 = mu_4
        self.transition_matrix = transition_matrix
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.N = N
        
    def calculate_mm1k_metrics(self, lambda_val, mu_val, K):
        """
        Calculate MM1K queue metrics with improved numerical stability
        """
        try:
            if mu_val == 0:
                return 0, 0, 0, 0, 1.0
                
            rho = lambda_val / mu_val
            
            # Case 1: rho ≈ 1
            if abs(rho - 1.0) < 1e-10:
                pi_k = 1 / (K + 1)
                P0 = pi_k
                L = K / 2
                
            # Case 2: rho ≈ 0
            elif abs(rho) < 1e-10:
                pi_k = 0
                P0 = 1.0
                L = 0
                
            # Case 3: normal case
            else:
                # Compute in log space for numerical stability
                log_rho = np.log(rho)
                
                # Check if rho > 1 (which can cause numerical issues)
                if rho > 1:
                    # For rho > 1, use asymptotic approximation
                    pi_k = K / (K + 1)
                    P0 = 1 / (K + 1)
                    L = K * (1 - 1/(K+1))
                    
                else:
                    # For rho < 1, use standard formulas but with careful computation
                    try:
                        # Compute π(K) carefully
                        if K * log_rho < -700:  # numpy.exp lowest value threshold
                            pi_k = 0
                        else:
                            # Use log-sum-exp trick for numerical stability
                            log_num = K * log_rho + np.log(max(1e-300, 1 - rho))
                            log_series = [i * log_rho for i in range(K + 2)]
                            max_log = max(log_series)
                            sum_exp = sum(np.exp(x - max_log) for x in log_series)
                            log_denom = max_log + np.log(sum_exp)
                            pi_k = np.exp(log_num - log_denom)
                        
                        # Compute P0 carefully
                        if (K + 1) * log_rho < -700:
                            P0 = 1 - rho
                        else:
                            P0 = (1 - rho) / (1 - np.exp((K + 1) * log_rho))
                        
                        # Compute L using a more stable formula
                        if K * log_rho < -700:
                            L = rho / (1 - rho)
                        else:
                            sum_n = 0
                            sum_rho_n = 0
                            for n in range(K + 1):
                                if n * log_rho > -700:
                                    rho_n = np.exp(n * log_rho)
                                    sum_n += n * rho_n
                                    sum_rho_n += rho_n
                            L = sum_n / sum_rho_n
                            
                    except (OverflowError, ZeroDivisionError):
                        # Fallback to approximations if computation fails
                        if rho < 1:
                            pi_k = 0
                            P0 = 1 - rho
                            L = rho / (1 - rho)
                        else:
                            pi_k = K / (K + 1)
                            P0 = 1 / (K + 1)
                            L = K * (1 - 1/(K+1))
    
            # Ensure values are within valid ranges
            pi_k = min(max(0, pi_k), 1)
            P0 = min(max(0, P0), 1)
            L = min(max(0, L), K)
            
            # Calculate derived metrics
            plr = pi_k  # Packet Loss Ratio = π(K)
            lambda_eff = lambda_val * (1 - plr)
            U = 1 - P0
            W = L / lambda_eff if lambda_eff > 0 else 0
            
            return lambda_eff, W, U, L, plr
            
        except Exception as e:
            print(f"Warning: Error occurred in MM1K calculation: {str(e)}")
            # Fallback to basic approximations
            if rho < 1:
                return lambda_val, 1/mu_val, rho, rho/(1-rho), 0
            else:
                plr = K/(K+1)
                return lambda_val * (1-plr), K/mu_val, 1.0, K, plr

    def calculate_mm1k_metrics_sim(self, lambda_val, mu_val, K, sim_time=100000, seed=None):
        """
        Calculate MM1K queue metrics using improved SimPy simulation
        """
        class MM1KQueue:
            def __init__(self, env, lambda_val, mu_val, K, seed=None):
                self.env = env
                self.lambda_val = lambda_val
                self.mu_val = mu_val
                self.K = K
                self.server = simpy.Resource(env, capacity=1)
                
                # Statistics
                self.queue_length = 0
                self.total_arrivals = 0
                self.total_losses = 0
                self.total_departures = 0
                self.area_queue_length = 0.0
                self.area_server_busy = 0.0
                self.last_event_time = 0
                self.waiting_times = []
                
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)

                # Start arrival process
                if lambda_val > 0:
                    env.process(self.arrival_process())
            
            def update_stats(self, current_time):
                time_since_last_event = current_time - self.last_event_time
                self.area_queue_length += self.queue_length * time_since_last_event
                self.area_server_busy += (1 if self.server.users else 0) * time_since_last_event
                self.last_event_time = current_time
            
            def arrival_process(self):
                while True:
                    # Generate next arrival
                    interarrival_time = np.random.exponential(1.0 / self.lambda_val)
                    yield self.env.timeout(interarrival_time)
                    
                    # Update statistics
                    self.update_stats(self.env.now)
                    self.total_arrivals += 1
                    
                    # Check if queue is full
                    if self.queue_length >= self.K:
                        self.total_losses += 1
                        continue
                    
                    self.queue_length += 1
                    
                    # Start service process
                    self.env.process(self.service_process())
            
            def service_process(self):
                arrival_time = self.env.now
                
                with self.server.request() as request:
                    yield request
                    
                    # Service time
                    service_time = np.random.exponential(1.0 / self.mu_val)
                    yield self.env.timeout(service_time)
                    
                    # Update statistics at departure
                    self.update_stats(self.env.now)
                    # 전체 시스템 체류 시간 계산 (대기 + 서비스)
                    self.waiting_times.append(self.env.now - arrival_time)  # 변경된 부분
                    self.queue_length -= 1
                    self.total_departures += 1

        # Create and run simulation
        env = simpy.Environment()
        queue = MM1KQueue(env, lambda_val, mu_val, K, seed)
        env.run(until=sim_time)
        
        # Calculate final metrics
        if queue.total_arrivals == 0:
            return 0, 0, 0, 0, 1.0

        # Update final statistics
        queue.update_stats(sim_time)
        
        # Calculate metrics
        plr = queue.total_losses / queue.total_arrivals if queue.total_arrivals > 0 else 0
        lambda_eff = lambda_val * (1 - plr)
        W = np.mean(queue.waiting_times) if queue.waiting_times else 0
        U = queue.area_server_busy / sim_time
        L = queue.area_queue_length / sim_time

        return lambda_eff, W, U, L, plr

    def analyze(self, use_simulation=False, sim_time=100000, seed=None):
        """
        Analyze the network using either theoretical calculations or simulation
        """
        # 나머지 코드는 동일하게 유지
        calc_method = self.calculate_mm1k_metrics_sim if use_simulation else self.calculate_mm1k_metrics
        lambda_0_base = self.lambda_0 * self.N
        
        # Server 1 (First queue)
        lambda_0_eff, W1, U1, L1, PLR1 = calc_method(lambda_0_base, self.mu_1, self.K1, sim_time, seed) if use_simulation else calc_method(lambda_0_base, self.mu_1, self.K1)
        
        # Server 2
        lambda_1_base = self.lambda_1 * self.N + lambda_0_eff * self.transition_matrix[0][1]
        lambda_1_eff, W2, U2, L2, PLR2 = calc_method(lambda_1_base, self.mu_2, self.K2, sim_time, seed) if use_simulation else calc_method(lambda_1_base, self.mu_2, self.K2)
        
        # Server 3
        lambda_2_base = self.lambda_2 * self.N + lambda_1_eff * self.transition_matrix[1][2]
        lambda_2_eff, W3, U3, L3, PLR3 = calc_method(lambda_2_base, self.mu_3, self.K3, sim_time, seed) if use_simulation else calc_method(lambda_2_base, self.mu_3, self.K3)
        
        # Server 4
        lambda_3_base = self.lambda_3 * self.N + lambda_2_eff * self.transition_matrix[2][3]
        lambda_3_eff, W4, U4, L4, PLR4 = calc_method(lambda_3_base, self.mu_4, self.K4, sim_time, seed) if use_simulation else calc_method(lambda_3_base, self.mu_4, self.K4)
        
        throughput = lambda_1_eff * (1 - self.transition_matrix[1][2]) + lambda_3_eff
        
        R_C = W1
        R_AH = W2 + W3 + W4
        
        return {
            'waiting_times': [W1, W2, W3, W4],
            'utilization': [U1, U2, U3, U4],
            'throughput': throughput,
            'response_times': [R_C, R_AH],
            'queue_lengths': [L1, L2, L3, L4],
            'effective_arrival_rates': [lambda_0_eff, lambda_1_eff, lambda_2_eff, lambda_3_eff],
            'packet_loss_ratios': [PLR1, PLR2, PLR3, PLR4]
        }

def run_scenarios(start_N=0, end_N=50, interval=5, use_simulation=False, sim_time=100000, seed=None):
    # 나머지 코드는 동일하게 유지
    base_params = {
        'lambda_0': 0.9, 
        'lambda_1': 55.99,
        'lambda_2': 0,
        'lambda_3': 0,
        'mu_1': 44.5,
        'mu_2': 3000.0,
        'mu_3': 25.0,
        'mu_4': 26.7,
        'transition_matrix': [[0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.015, 0.0],
                             [0.0, 0.0, 0.0, 0.1],
                             [0.0, 0.0, 0.0, 0.0]],
        'K1': 100,
        'K2': 10000,
        'K3': 100,
        'K4': 100,
        'N': 0
    }

    scenarios = {}
    for N in range(start_N, end_N + interval, interval):
        scenario_name = str(N)
        scenarios[scenario_name] = {**base_params, 'N': N}

    results = {}
    for scenario_name, params in scenarios.items():
        network = JacksonNetworkMM1K(**params)
        results[scenario_name] = network.analyze(use_simulation=use_simulation, 
                                              sim_time=sim_time, 
                                              seed=seed)
        print(f"Scenario {scenario_name} completed")

    return results  # 결과를 반환하도록 수정

if __name__ == "__main__":
    # Run both theoretical and simulation scenarios
    print("Running theoretical analysis...")
    theoretical_results = run_scenarios(start_N=10, end_N=30, interval=1, use_simulation=False)
    
    print("\nRunning simulation analysis...")
    simulation_results = run_scenarios(start_N=10, end_N=30, interval=1, use_simulation=True, sim_time=1000, seed=42)

    print("\nDone")
    
    # 결과 비교를 위한 출력 추가
#    plot_theory_vs_simulation(theoretical_results, simulation_results)
