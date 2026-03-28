import matplotlib.pyplot as plt

def component_energy(power, time):
    """
    Calculates energy of a component
    power = Watts, time = seconds
    returns Joules
    """
    return power * time

# --- POWER VALUES (Watts) ---
P_fpga = 3.2      # FPGA SNN computation
P_memory = 0.8    # Memory power
P_io = 0.3        # I/O communication power
P_host = 1.5      # Host CPU power

# --- EXECUTION TIMES (seconds) ---
T_preprocess = 0.04   # EEG preprocessing
T_encoding = 0.03     # EEG-to-spike encoding
T_snn = 0.08          # SNN inference on FPGA
T_memory = 0.08       # Memory access during inference
T_io = 0.01           # Data transfer
T_host = 0.02         # Host side computation

# --- SPIKING INFORMATION ---
N_spikes = 1200
inferences_per_character = 5

# --- ENERGY CALCULATIONS ---
E_preprocess = component_energy(P_host, T_preprocess)
E_encoding = component_energy(P_host, T_encoding)
E_snn = component_energy(P_fpga, T_snn)
E_memory = component_energy(P_memory, T_memory)
E_io = component_energy(P_io, T_io)
E_host = component_energy(P_host, T_host)

E_total = E_preprocess + E_encoding + E_snn + E_memory + E_io + E_host

E_per_spike = E_snn / N_spikes
E_per_character = E_total * inferences_per_character
E_per_inference = E_total

# --- DISPLAY RESULTS ---
print("\n" + "="*20)
print("ENERGY RESULTS")
print("="*20)
print(f"Preprocessing Energy:  {E_preprocess:.6f} J")
print(f"Spike Encoding Energy: {E_encoding:.6f} J")
print(f"SNN FPGA Energy:       {E_snn:.6f} J")
print(f"Memory Energy:         {E_memory:.6f} J")
print(f"IO Energy:             {E_io:.6f} J")
print(f"Host Energy:           {E_host:.6f} J")
print("-" * 20)
print(f"Total Energy per Inference:  {E_per_inference:.6f} J")
print(f"Energy per Spike:            {E_per_spike:.10f} J")
print(f"Energy per Character Typed:  {E_per_character:.6f} J")

# --- LATENCY CHECK ---
total_latency = T_preprocess + T_encoding + T_snn + T_io + T_host

print(f"\nTotal System Latency: {total_latency*1000:.3f} ms")

if total_latency <= 0.3:
    print("Latency Requirement Met (<300 ms)")
else:
    print("Latency Requirement NOT Met")

# --- ENERGY BREAKDOWN VISUALIZATION ---
labels = ['Preprocessing', 'Spike Encoding', 'SNN FPGA', 'Memory', 'IO', 'Host']
energy_components = [E_preprocess, E_encoding, E_snn, E_memory, E_io, E_host]

plt.figure(figsize=(10, 6))
plt.bar(labels, energy_components, color='teal')
plt.ylabel('Energy (Joules)')
plt.title('Energy Consumption per Component')
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- ENERGY PER TIME WINDOW (REAL-TIME OPERATION) ---
time_windows = list(range(1, 51))
energy_runtime = [E_total * w for w in time_windows]

plt.figure(figsize=(10, 6))
plt.plot(time_windows, energy_runtime, linewidth=2, color='darkorange')
plt.xlabel('Number of Inference Windows')
plt.ylabel('Total Energy (J)')
plt.title('Energy Consumption Over Time')
plt.grid(True)
plt.show()
# --- SOPs CALCULATION (Synaptic Operations Per Second) ---

# Assumption: synapses (fan-out) per spike
synapses_per_spike = 100   # adjust based on your SNN architecture

# Total synaptic operations
total_sops = N_spikes * synapses_per_spike

# SOPs = synaptic operations / time
SOPs = total_sops / T_snn

print("\n" + "="*20)
print("SOPs CALCULATION")
print("="*20)
print(f"Synapses per Spike:       {synapses_per_spike}")
print(f"Total Synaptic Ops:       {total_sops}")
print(f"Effective SOPs:           {SOPs:.2f} SOPs")
print(f"Effective MSOPs:          {SOPs/1e6:.6f} MSOPs")
print("="*20)
