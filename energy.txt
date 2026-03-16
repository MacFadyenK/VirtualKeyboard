clc;
clear;
close all;


% EEG-Driven SNN Virtual Keyboard
% Energy Evaluation Script



% POWER VALUES (Watts)

% Replace with measured values from FPGA tools if available

P_fpga = 3.2;        % FPGA SNN computation
P_memory = 0.8;      % Memory power
P_io = 0.3;          % I/O communication power
P_host = 1.5;        % Host CPU power



% EXECUTION TIMES (seconds)

% These should satisfy the system latency requirement (<300 ms)

T_preprocess = 0.04;   % EEG preprocessing
T_encoding = 0.03;     % EEG-to-spike encoding
T_snn = 0.08;          % SNN inference on FPGA
T_memory = 0.08;       % Memory access during inference
T_io = 0.01;           % Data transfer
T_host = 0.02;         % Host side computation



% SPIKING INFORMATION


N_spikes = 1200;           % Total spikes during inference
inferences_per_character = 5;



% ENERGY CALCULATIONS


E_preprocess = component_energy(P_host, T_preprocess);
E_encoding = component_energy(P_host, T_encoding);
E_snn = component_energy(P_fpga, T_snn);
E_memory = component_energy(P_memory, T_memory);
E_io = component_energy(P_io, T_io);
E_host = component_energy(P_host, T_host);

E_total = E_preprocess + E_encoding + E_snn + E_memory + E_io + E_host;

E_per_spike = E_snn / N_spikes;
E_per_character = E_total * inferences_per_character;
E_per_inference = E_total;



% DISPLAY RESULTS


fprintf("\nENERGY RESULTS\n");

fprintf("Preprocessing Energy: %.6f J\n", E_preprocess);
fprintf("Spike Encoding Energy: %.6f J\n", E_encoding);
fprintf("SNN FPGA Energy: %.6f J\n", E_snn);
fprintf("Memory Energy: %.6f J\n", E_memory);
fprintf("IO Energy: %.6f J\n", E_io);
fprintf("Host Energy: %.6f J\n", E_host);


fprintf("Total Energy per Inference: %.6f J\n", E_per_inference);
fprintf("Energy per Spike: %.10f J\n", E_per_spike);
fprintf("Energy per Character Typed: %.6f J\n", E_per_character);




% LATENCY CHECK


total_latency = T_preprocess + T_encoding + T_snn + T_io + T_host;

fprintf("\nTotal System Latency: %.3f ms\n", total_latency*1000);

if total_latency <= 0.3
    fprintf("Latency Requirement Met (<300 ms)\n");
else
    fprintf("Latency Requirement NOT Met\n");
end



% ENERGY BREAKDOWN VISUALIZATION

 energy_components = [
    E_preprocess
    E_encoding
    E_snn
    E_memory
    E_io
    E_host
];

labels = {
    'Preprocessing'
    'Spike Encoding'
    'SNN FPGA'
    'Memory'
    'IO'
    'Host'
};

figure
bar(energy_components)
set(gca,'XTickLabel',labels)
ylabel('Energy (Joules)')
title('Energy Consumption per Component') 
grid on



% ENERGY PER TIME WINDOW (REAL-TIME OPERATION)


time_windows = 1:50;
energy_runtime = E_total * time_windows;

figure
plot(time_windows, energy_runtime, 'LineWidth',2)
xlabel('Number of Inference Windows')
ylabel('Total Energy (J)')
title('Energy Consumption Over Time')
grid on



% FUNCTION DEFINITIONS


function E = component_energy(power, time)

% Calculates energy of a component
% power = Watts
% time = seconds
% energy output = Joules

E = power * time;

end