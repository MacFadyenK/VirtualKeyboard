import matplotlib.pyplot as plt
import numpy as np


def component_energy():

    # -----------------------------
    # CONFIGURATION
    # -----------------------------
    load_levels = np.linspace(0.5, 1.5, 10)   # GENERAL ALGORITHM REPRESENTING LOW, NOMINAL, HIGH LOADS/SYSTEM-STATES
    N_inferences = 100                        # number of inferences per load

    # Baseline powers (Watts)
    P_fpga_base   = 1.0
    P_memory_base = 0.5
    P_io_base     = 0.2
    P_host_base   = 3.0

    # Baseline times (seconds)
    T_preprocess_base = 0.04
    T_encoding_base   = 0.03
    T_snn_base        = 0.05
    T_memory_base     = 0.04
    T_io_base         = 0.01
    T_host_base       = 0.02

    N_spikes = 1200
    inferences_per_character = 5

    # -----------------------------
    # STORAGE FOR ALL LOAD LEVELS
    # -----------------------------
    avg_results = {
        "load": [],
        "avg_pre": [],
        "avg_enc": [],
        "avg_snn": [],
        "avg_mem": [],
        "avg_io": [],
        "avg_host": [],
        "avg_total": []
    }
    # Lists for this load level
    E_preprocess = []
    E_encoding = []
    E_snn = []
    E_memory = []
    E_io = []
    E_host = []
    E_total = []
    FLOPS_list = []
    SOPS_list = []
    # -----------------------------
    # SWEEP OVER LOAD LEVELS
    # -----------------------------
    for load in load_levels:
        # -----------------------------
        # RUN MULTIPLE INFERENCES
        # -----------------------------
        for _ in range(N_inferences):
            # Firing rate(spikes/s) can be scaled with load if needed, e.g.:
            firing_rate = 1000 * (1.0 + 0.5 * (load - 1))  # increase by up to 50% at high load

            # Scale power with load
            P_fpga  = P_fpga_base   * load
            P_memory = P_memory_base * (0.8 + 0.2 * load)
            P_io     = P_io_base
            P_host   = P_host_base   * (0.9 + 0.1 * load)

            # Scale time with load
            T_preprocess = T_preprocess_base
            T_encoding   = T_encoding_base
            T_snn        = T_snn_base * (1.0 + 0.3 * (load - 1))
            T_memory     = T_memory_base * (1.0 + 0.2 * (load - 1))
            T_io         = T_io_base
            T_host       = T_host_base

            # Energy calculations
            e_pre = P_host * T_preprocess
            e_enc = P_host * T_encoding
            e_snn = P_fpga * T_snn
            e_mem = P_memory * T_memory
            e_io  = P_io * T_io
            e_host = P_host * T_host

            total = e_pre + e_enc + e_snn + e_mem + e_io + e_host

            # Store
            E_preprocess.append(e_pre)
            E_encoding.append(e_enc)
            E_snn.append(e_snn)
            E_memory.append(e_mem)
            E_io.append(e_io)
            E_host.append(e_host)
            E_total.append(total)

        # -----------------------------
        # FLOPS & SOPS CALCULATION (per load)
        # -----------------------------
        ops_per_spike = 2
        total_ops = N_spikes * ops_per_spike

        # Use the same T_snn formula you used during the loop
        avg_T_snn = T_snn_base * (1.0 + 0.3 * (load - 1))

        FLOPS = total_ops / avg_T_snn
        SOPS = avg_T_snn * FLOPS * firing_rate
        

        # Store FLOPS and SOPS for this load level
        FLOPS_list.append(FLOPS)
        SOPS_list.append(SOPS)

        print("\n" + "="*20)
        print("FLOPS CALCULATION")
        print("="*20)
        print(f"Operations per Spike:     {ops_per_spike}")
        print(f"Total Operations:         {total_ops}")
        print(f"Effective FLOPS:          {FLOPS:.2f} FLOPS")
        print(f"Effective SOPS:           {SOPS/1e6:.6f} SOPS")
        print("="*20)

        # -----------------------------
        # AVERAGES FOR THIS LOAD LEVEL
        # -----------------------------
        avg_pre = np.mean(E_preprocess)
        avg_enc = np.mean(E_encoding)
        avg_snn = np.mean(E_snn)
        avg_mem = np.mean(E_memory)
        avg_io  = np.mean(E_io)
        avg_host = np.mean(E_host)
        avg_total = np.mean(E_total)

        # Save results
        avg_results["load"].append(load)
        avg_results["avg_pre"].append(avg_pre)
        avg_results["avg_enc"].append(avg_enc)
        avg_results["avg_snn"].append(avg_snn)
        avg_results["avg_mem"].append(avg_mem)
        avg_results["avg_io"].append(avg_io)
        avg_results["avg_host"].append(avg_host)
        avg_results["avg_total"].append(avg_total)

        # -----------------------------
        # PRINT SUMMARY FOR THIS LOAD
        # -----------------------------
        print("\n" + "="*50)
        print(f"LOAD LEVEL: {load:.2f}")
        print("="*50)
        print(f"Avg Preprocessing Energy:   {avg_pre:.6f} J")
        print(f"Avg Encoding Energy:        {avg_enc:.6f} J")
        print(f"Avg SNN FPGA Energy:        {avg_snn:.6f} J")
        print(f"Avg Memory Energy:          {avg_mem:.6f} J")
        print(f"Avg IO Energy:              {avg_io:.6f} J")
        print(f"Avg Host Energy:            {avg_host:.6f} J")
        print("-"*50)
        print(f"Avg Total Energy/Inference: {avg_total:.6f} J")

    # -----------------------------
    # FINAL PLOT: ENERGY VS LOAD
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(avg_results["load"], avg_results["avg_total"], marker='o', linewidth=2)
    plt.xlabel("Hardware Load Level")
    plt.ylabel("Average Energy per Inference (J)")
    plt.title("Energy Scaling with Hardware Load")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Labels and corresponding average energy values
    categories = ['Preprocess', 'Encoding', 'SNN', 'Memory', 'IO', 'Host']
    avg_energy_values = [avg_pre, avg_enc, avg_snn, avg_mem, avg_io, avg_host]

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(categories, avg_energy_values, color='teal')

    # Axis labels and title
    plt.ylabel("Average Energy (J)")
    plt.title("Average Energy Consumption per Component")

    # Optional styling
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15)
    plt.tight_layout()

    # Display the plot
    plt.show()

    # -----------------------------
    # AVERAGE FLOPS & SOPS ACROSS LOADS
    # -----------------------------
    avg_FLOPS = np.mean(FLOPS_list)

    print("\n" + "="*40)
    print("AVERAGE FLOPS & SOPS ACROSS LOAD LEVELS")
    print("="*40)
    print(f"Average FLOPS:   {avg_FLOPS:.2f}")
    print(f"Average SOPS:  {np.mean(SOPS_list)/1e6:.6f}")
    print("="*40)


    return avg_results

component_energy()
