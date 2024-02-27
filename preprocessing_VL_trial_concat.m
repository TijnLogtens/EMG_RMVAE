%% Marshall test pre-processing
% Simple script to select the data we need from the segmented trials and
% trial concatenate them, we then apply a Hann window at 400 ms and export
% the result to CSV.

%% Load data into seperate variables
% Define file name
name = ['VL_RAW/S6_plat_comp/S6_PiDe_20percent_onlyPlateau_window_'];
num_trials = 4;

% Create array of filenames
fileList = {};
for j = 1:num_trials
    fileList{end+1} = [name, num2str(j),'.mat'];
end

% Initialize a cell array to store loaded data
loadedData = cell(1, numel(fileList));

% Loop through the file list and load data into objects
for i = 1:numel(fileList)
    % Generate a variable name
    variableName = ['data', num2str(i)];
    
    % Load the data from the .mat file into a dynamically named variable
    loadedData{i} = load(fileList{i});
    
    % Assign the loaded data to a variable with the generated name
    eval([variableName ' = loadedData{i};']);
end

%% Move the spike trains to their own matrices
% We want to build the n matrices that just contain the raster binary spike
% trains.
for k = 1:num_trials
    % Generate a variable name
    tmp = ['data', num2str(k)];

    MUs = eval([tmp '.edition.binarySpikeTrains(1,:)']);
    
    % Move MUs into same matrix
    MUs_reshaped = zeros(1, length(MUs{1,2}(1,:)));
    
    for m = 3:length(MUs)
        for n = 1:size(MUs{1,m},1)
            MUs_reshaped = [MUs_reshaped; MUs{1,m}(n,:)];
        end
    end

    % Define output variables
    spikeTrainVar = ['spikeTrains', num2str(k)];
    eval([spikeTrainVar ' = MUs_reshaped;']);
end 

%% Remove MUs that don't spike for windows of 400 ms
% Combine the matrices into a 3D array
all_trials = [spikeTrains1, spikeTrains2, spikeTrains3, spikeTrains4];

% Set the threshold for the number of consecutive zeros
consecutive_zeros_threshold = int32(0.4*2048); % 0.4 s of data

% Initialize a logical mask for all trials combined
mask_combined = false(size(all_trials, 1), 1);

% Iterate through each row to find rows with more than consecutive_zeros_threshold consecutive zeros
for i = 1:size(all_trials, 1)
    row_combined = all_trials(i, :);
    if nnz(conv(double(row_combined == 0), ones(1, consecutive_zeros_threshold), 'valid') == consecutive_zeros_threshold)
        mask_combined(i) = true;
    end
end

% Apply logical mask to remove corresponding rows from all trials combined
all_trials_filtered = all_trials(~mask_combined, :);

% Split the filtered trials back into separate matrices
num_cols_per_trial1 = size(spikeTrains1, 2);
num_cols_per_trial2 = size(spikeTrains2, 2);
num_cols_per_trial3 = size(spikeTrains3, 2);

spikeTrains1 = all_trials_filtered(:, 1:num_cols_per_trial1);
spikeTrains2 = all_trials_filtered(:, num_cols_per_trial1+1 : num_cols_per_trial1+num_cols_per_trial2);
spikeTrains3 = all_trials_filtered(:, num_cols_per_trial1+num_cols_per_trial2+1 : num_cols_per_trial1+num_cols_per_trial2+num_cols_per_trial3);
spikeTrains4 = all_trials_filtered(:, end - size(spikeTrains4, 2) + 1 : end);

%% Visualize spike trains
figure;
hold on;

% Loop through each neuron and plot its spike times
[num_neurons, num_time_steps] = size(spikeTrains1);
time_vector = 1:num_time_steps-30000;

for neuron = 1:num_neurons
    spikes = find(spikeTrains1(neuron, 1:4000)); % Find non-zero elements
    scatter(spikes, neuron * ones(1, length(spikes)), 'k.', 'MarkerFaceColor', 'k');
end
xlabel('Time');
ylabel('MUs');
title('S1 Plateau Motor Unit Spike Trains');
hold off;


%% Calculate spiking rates for every set of spike trains
% Define general parameters
window_size = 0.4; % Size window Hann filter in seconds
sampling_rate = 2048; % Sampling rate in Hz
min_max = true; 

% Define trial based parameters
for o = 1:num_trials
    cur_spikeTrain = eval(['spikeTrains', num2str(o)]);

    duration = size(cur_spikeTrain, 2)/sampling_rate; % Duration of the data in seconds
    time_vector = linspace(0, duration, sampling_rate * duration); % vector of length samples
    
    % Define Hanning window for smooting
    HanningW = 2/round(sampling_rate*window_size)*hann(round(sampling_rate*window_size)); 
   
    % Get spike train dimensions
    num_neurons = size(cur_spikeTrain, 1);
    num_samples = size(cur_spikeTrain, 2);
    
    % Define firing rates vector
    firing_rates = zeros(num_neurons, length(time_vector));
    
    % Apply kernel
    for neuron = 1:num_neurons
        firing_rates(neuron, :) = filtfilt(HanningW,1,cur_spikeTrain(neuron, :)*sampling_rate); % Firings is the binary spike train
    end
    
    % Remove first and last 2 seconds
    two_seconds = sampling_rate * 2;
    firing_rates = firing_rates(:, two_seconds + 1:num_samples-two_seconds);    
    
    if min_max
        % Normalise
        [num_neurons, num_samples] = size(firing_rates);
        
        % normalise individual rates
        for i = 1:num_neurons
            % Get current rates
            rate = firing_rates(i,:);
        
            % find min and max
            min_val = min(rate);
            max_val = max(rate);
            
            div = max_val - min_val;
            if (div == 0)
                normalised_rate = rate*0;
            else
                % min-max normalise
                normalised_rate = (rate - min_val)/div;
            end
    
            % update firing rates
            firing_rates(i, :) = normalised_rate;
        end
    end

    det_firing_rates = detrend(firing_rates,1);

    % Define output variables
    firingRateVar = ['firingRate', num2str(o)];
    eval([firingRateVar ' = firing_rates;']);

    detFiringRateVar = ['detFiringRate', num2str(o)];
    eval([detFiringRateVar ' = det_firing_rates;']);
end 

%% Visualise firing rates
% Create a plot with overlapping firing rate time series
figure;
time_points = 1:size(firingRate1, 2);

for i = 1:num_neurons
    plot(time_points, firingRate1(i, :) + (i-1), 'LineWidth', 1);
    hold on;
end

xlabel('Time');
ylabel('MUs');
title('Trial 1 MU Firing Rates Over Time');
hold off;


%% Aligning the different trials (Shifting)
% We now want to align the different trials so we can get the most accurate
% aligned averaged firing rates. To do this in the best way possible we
% will want to align them based the minismised MSE between the different
% trials. Here we do so by shifting the trials compared to eachother over a
% 200 ms interval.

% Create a cell array to store the firing rate matrices
firing_rate_matrices = {firingRate1, firingRate2, firingRate3, firingRate4}; % }; %

% Find the minimum length among all trials
min_trial_length = min(cellfun(@(x) size(x, 2), firing_rate_matrices));

% Create an empty matrix to store the aligned and averaged firing rates
aligned_firing_rates = cell(size(firing_rate_matrices));

% Set the maximum shift amount
maxShift = 200;

for i = 1:numel(firing_rate_matrices)
    currentMatrix = firing_rate_matrices{i};
    
    % Find the optimal shift within the constrained range
    [~, shiftAmount] = max(corr(currentMatrix(:, 1:min_trial_length)', firing_rate_matrices{1}(:, 1:min_trial_length)'));
    shiftAmount = min(max(-maxShift, shiftAmount), maxShift);
    
    % Apply the shift and slice
    aligned_firing_rates{i} = currentMatrix(:, max(1, shiftAmount):(min(size(currentMatrix, 2), shiftAmount + min_trial_length)));
end

% Step 4: Concatenate matrices
final_firing_rates = cat(2, aligned_firing_rates{:});

% remove first row
final_firing_rates(1,:) = [];

%% Visualise concatenated firing rates
% Create a plot with overlapping firing rate time series
figure;
time_points = 1:size(final_firing_rates, 2);

for i = 1:num_neurons-1
    plot(time_points, final_firing_rates(i, :) + (i-1), 'LineWidth', 1);
    hold on;
end

xlabel('Time');
ylabel('MUs');
title('Concatenated MU Firing Rates Over Time');
hold off;

%% Export to csv
% Add path
path_comp = 'Trial_concat_tot/';
file_name = 'S6_VL_plat_concat';

% Export
writematrix(final_firing_rates, [path_comp, file_name, '.csv']);