%Left comments below for those who want to understand preprocessing
%progress so far

% Load the .mat file into MATLAB workspace
data = load('C:\Users\MikeK\Downloads\s17.mat');

% Allows us to see what is contained inside dataset
fieldnames(data)

% How many training runs we have
size(data.train)

% Selecting dataset, we can choose from 1 & 2
t2 = data.train{1};
% t2 = data.train{2};

unique(t2.markers_target)

% lets us see where target stimulus and where non target stim is
target_indices = find(t2.markers_target == 1);
nontarget_indices = find(t2.markers_target == 2);

% define parameters, so sampling rate (512hz) and epoch length (600ms)
samplingrate = t2.srate;
epoch_length = round(0.6*samplingrate);

% Design bandpass filter (0.5–10 Hz)
[b,a] = butter(4, [0.5 10]/(samplingrate/2), 'bandpass');


% Use raw EEG data
raw_data = t2.data;

% Pick first target stimulus
firstargetstimulus = target_indices(1);

% Extract 600 ms window after stimulus
one_trial = raw_data(:, firstargetstimulus : firstargetstimulus + epoch_length - 1);

pz_index = 13;   % example electrode, we can choose many but a paper mentions Pz/13 is strongest

time = (0:epoch_length-1) / samplingrate * 1000;  % convert to milliseconds



% Extract target epochs (raw, no filter yet)
raw_data = t2.data;

% lets us create these arrays, channels, samples, and number of trials we
target_epochs = zeros(32, epoch_length, length(target_indices));
nontarget_epochs = zeros(32, epoch_length, length(nontarget_indices));

% Extracts all target trials
for i = 1:length(target_indices)
    firstargetstimulus = target_indices(i);
    target_epochs(:,:,i) = raw_data(:, firstargetstimulus:firstargetstimulus+epoch_length-1);
end

% extracts all nontarget trials
for i = 1:length(nontarget_indices)
    firstargetstimulus = nontarget_indices(i);
    nontarget_epochs(:,:,i) = raw_data(:, firstargetstimulus:firstargetstimulus+epoch_length-1);
end

% Average across trials
average_target = mean(target_epochs,3);
average_nontarget = mean(nontarget_epochs,3);


time = (0:epoch_length-1)/samplingrate*1000;

one_trial_filtered = filtfilt(b,a, one_trial')';


figure;
plot(time, one_trial(pz_index,:), 'k');
hold on;
plot(time, one_trial_filtered(pz_index,:), 'r');

legend('Raw','Filtered');
title('Single Trial: Raw vs Bandpass Filtered');



