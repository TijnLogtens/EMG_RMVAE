%% Data processing GM
% Turn the chunky Francois processed GM data into the same format that the
% windowed data had.

%% Load data
% Define data file name
name_data = 'GM RAW/S7_GM_LeCl/S7_LECl_20percent.mat';

% Load matrix
data = load(name_data);

% Define time_points name
name_sep = 'GM RAW/S7_GM_LeCl/S7_LECl_20percent_selected_windows_timepoints.mat';

% Load separation matrix 
time_sep = load(name_sep);
num_windows = length(time_sep.limOfEachWindow(:,1));

%% Convert spike indices to binary spike trains
% get MUs
MUs = data.edition.Distimeclean(1,:);

% Flatten spike times to 2D
MU_spike_times = cat(2, MUs{:})';

% Find the maximum length of arrays
max_spikes = max(cellfun(@numel, MU_spike_times));

% Initialize a matrix with NaN values
MU_spikes = NaN(numel(MU_spike_times), max_spikes);

% Fill the matrix with the arrays, padding with NaN where needed
for i = 1:numel(MU_spike_times)
    current_array = MU_spike_times{i};
    MU_spikes(i, 1:numel(current_array)) = current_array;
end

% Build the binary spike trains
% Find the maximum value in the neuron data
final_spike = max(MU_spikes(:));

% Initialize a binary spike train matrix with zeros
binary_spike_trains = zeros(size(MU_spikes, 1), final_spike);

% Iterate through each neuron
for i = 1:size(MU_spikes, 1)
    % Get current row
    inds = MU_spikes(i, :);
    % Set corresponding indices to 1 in the binary spike train matrix
    binary_spike_trains(i, inds(inds <= final_spike)) = 1;
end

%% Identify the separate windows
for n = 1:num_windows
    p1 = time_sep.limOfEachWindow(n, 1);
    p2 = time_sep.limOfEachWindow(n, 2);
    
    if p2 > size(binary_spike_trains,2)
        p2 = size(binary_spike_trains,2);
    end

    window = binary_spike_trains(:, p1:p2);
    
    % Output separated window
    window_string = ['window_', num2str(n)];
    eval([window_string ' = window;']);
end

%% Remove MUs that don't spike for windows of 400 ms
% Combine the matrices into a 3D array
all_trials = [window_1, window_2, window_3]; %, window_4];

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
num_cols_per_trial1 = size(window_1, 2);
num_cols_per_trial2 = size(window_2, 2);
num_cols_per_trial3 = size(window_3, 2);

window_1 = all_trials_filtered(:, 1:num_cols_per_trial1);
window_2 = all_trials_filtered(:, num_cols_per_trial1+1 : num_cols_per_trial1+num_cols_per_trial2);
window_3 = all_trials_filtered(:, num_cols_per_trial1+num_cols_per_trial2+1 : num_cols_per_trial1+num_cols_per_trial2+num_cols_per_trial3);
% window_4 = all_trials_filtered(:, end - size(window_4, 2) + 1 : end);


%% Calculate firing rates for every set of continuous spike trains
% Define general parameters
window_size = 0.4; % Size window Hanning filter in seconds
sampling_rate = 2048; % Sampling rate in Hz

% Define trial based parameters
for o = 1:num_windows
    cur_spikeTrain = eval(['window_', num2str(o)]);

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

    % Define output variables
    firingRateVar = ['firingRate', num2str(o)];
    eval([firingRateVar ' = firing_rates;']);
end 

%% Align and concatenate the correct windows

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

%% Detrend firing rates
det_final_firing_rates = detrend(final_firing_rates, 1);

%% Plot the firing rates
figure;
plot(final_firing_rates');

%% Export to csv
% Add path
path_comp = 'Trial_concat_tot/';
file_name = 'S7_GM_plat_concat';

% Export
writematrix(final_firing_rates, [path_comp, file_name, '.csv']);
