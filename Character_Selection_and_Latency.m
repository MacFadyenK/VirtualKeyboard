fs = 256;  
%Example EEG Data

n_samples = 100; 
% Example Number of Samples

time_points = (0:n_samples-1) / fs;  
%Example time points

eeg_data = rand(n_samples, 4);
%Sumulate EEG data for 4 channels

eeg_features = mean(eeg_data, 1);
%Mean of each channel

disp('EEG Features (Mean of 4 Channels):');
disp(eeg_features);

prob_matrix = rand(2, 4);     %Random values between 0 and 1
prob_matrix = prob_matrix / sum(prob_matrix(:));   %Normalize to sum to 1

disp('Simulated Probability Matrix:');
disp(prob_matrix);

row_probabilities = sum(prob_matrix, 2);      %Sum each row
coulumn_probabilities = sum(prob_matrix, 1);    % Sum each column

[maxRowValue, indexrow] = max(row_probabilities);
[maxColValue, indexcol] = max(coulumn_probabilities);
%Finds the the maximum of row and column

letters = {'A', 'E', 'H', 'N', 'O', 'R', 'S', 'T'};
%Maps out matrix of the letters

selected_char = letters{(indexrow-1)*4 + indexcol};
% For the 2x4 grid: first row = 1-4 and second row = 5-8

disp(['Selected Character: ', selected_char]);
%Displays the given letter







%EEG - Latency Example Code

clear;
clc;

%Example EEG window (8 channels, 256 samples)
eeg_window = rand(8, 256);

%Start total latency timer
tic;

%Preprocessing
t1 = tic;
processed = (eeg_window - mean(eeg_window, 2)) ./ std(eeg_window, 0, 2);
preprocess_time = toc(t1);

% Spike Encoding (Measures the time it turned continuous EEG signals into
% spikes)
t2 = tic;
spike_train = processed > 0.5;
encoding_time = toc(t2);

% SNN Inference
t3 = tic;
weights = rand(8, 8);
snn_output = weights * mean(spike_train, 2);
snn_time = toc(t3);

% Decision Logic
t4 = tic;
[~, predicted_key] = max(snn_output);
decision_time = toc(t4);

total_latency = toc;

fprintf('Total Latency: %.4f ms\n', total_latency * 1000);
fprintf('Preprocessing Time: %.4f ms\n', preprocess_time * 1000);
fprintf('Encoding Time: %.4f ms\n', encoding_time * 1000);
fprintf('SNN Time: %.4f ms\n', snn_time * 1000);
fprintf('Decision Time: %.4f ms\n', decision_time * 1000);



% Energy calculation

%Example Power Draw (W) (Will need to know CPU/GPU average draw)
average_power = 20;

energy = average_power * total_latency;

fprintf('Energy per Inference: %.6f Joules\n', energy);