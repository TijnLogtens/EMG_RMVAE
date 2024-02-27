# Imports for this project
import statistics
from collections import defaultdict
from tensorflow import keras
from keras.layers import Input, Dense, Lambda, LeakyReLU
from keras.models import Model
from keras.constraints import NonNeg
import keras.backend as K
import numpy as np
import pandas as pd
from os.path import exists


# Rectified Monotonic Variational Autoencoder
# Default non-linear vae
def build_vae(input_dim, latent_dim):
    # Encoder
    encoder_inputs = Input(shape=(input_dim,))
    encoder_hidden = Dense(128, activation='relu', kernel_constraint=NonNeg())(encoder_inputs)
    z_mean = Dense(latent_dim)(encoder_hidden)
    z_log_var = Dense(latent_dim)(encoder_hidden)

    # Temperature parameter
    temperature = 0.9  # The close to 1 the more it promotes high variance sampling

    # Sampling from the latent space with temperature scaling
    def sampling(args):
        z_mean, z_log_var = args
        batch_size = K.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        scaled_epsilon = epsilon / temperature  # Scale epsilon by the temperature
        return z_mean + K.exp(0.5 * z_log_var) * scaled_epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_inputs = Input(shape=(latent_dim,))
    decoder_hidden = Dense(128, activation='relu', kernel_constraint=NonNeg())(decoder_inputs)
    decoder_outputs = Dense(input_dim)(decoder_hidden)
    decoder_outputs = LeakyReLU(alpha=0.05)(decoder_outputs)

    # Build VAE model
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = Model(encoder_inputs, vae_outputs, name="vae")

    # Compile VAE model
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
    vae.compile(optimizer=optimizer, loss='mse')

    return vae


# Default linear RMVAE
def build_linear_vae(input_dim, latent_dim):
    # Encoder
    encoder_inputs = Input(shape=(input_dim,))
    encoder_hidden = Dense(128, activation='linear', kernel_constraint=NonNeg())(encoder_inputs)
    z_mean = Dense(latent_dim)(encoder_hidden)
    z_log_var = Dense(latent_dim)(encoder_hidden)

    # Temperature parameter
    temperature = 0.9  # The close to 1 the more it promotes high variance sampling

    # Sampling from the latent space with temperature scaling
    def sampling(args):
        z_mean, z_log_var = args
        batch_size = K.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        scaled_epsilon = epsilon / temperature  # Scale epsilon by the temperature
        return z_mean + K.exp(0.5 * z_log_var) * scaled_epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_inputs = Input(shape=(latent_dim,))
    decoder_hidden = Dense(128, activation='linear', kernel_constraint=NonNeg())(decoder_inputs)
    decoder_outputs = Dense(input_dim, activation='linear')(decoder_hidden)

    # Build VAE model
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = Model(encoder_inputs, vae_outputs, name="vae")

    # Compile VAE model
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
    vae.compile(optimizer=optimizer, loss='mse')

    return vae


def row_has_high_zeros(row, threshold):
    return (row == 0).sum() < threshold


# Define cut off value for percentage allowed zeroes per MU
threshold_percentage = 90

# Number of repetitions of model fit
num_iterations = 10

# Model parameters
epochs = 60
batch_size = 32
latent_dim = 1

# Loop variables
subjects = ["S1_VL", "S2_VL", "S3_VL", "S4_VL", "S5_VL", "S6_VL", "S1_GM", "S2_GM", "S3_GM", "S4_GM", "S6_GM", "S7_GM"]
conditions = ["plat", "sin025", "sin1", "sin3"]
path = "/home/user/"

# Result storage
cv = []

# Loop over conditions
for s in range(len(subjects)):
    rank_cur_con = []

    # Loop over subjects
    for c in range(len(conditions)):
        # Load data
        f1 = path + "Trial_concat_tot/" + subjects[s] + "_" + conditions[c] + "_concat.csv"

        # Check if file exists (not all GM have sin3 MUs)
        if not exists(f1):
            continue

        # Load subject data
        data_in = pd.read_csv(f1, header=None)
        print(f"{subjects[s]}, condition {conditions[c]}")

        # Split into train and test trials (if plat/up trials = 4 else trials = 3)
        if conditions[c] == "plat" or conditions[c] == "up":
            n_trials = 4
        else:
            n_trials = 3

        # Remove 0 rows due to terrifying results (and dividing by zero problems)
        # Calculate the threshold for the number of zeros
        threshold = threshold_percentage / 100 * len(data_in.columns)

        # Apply the function to each row and use boolean indexing to filter out rows
        data_in = data_in[data_in.apply(lambda row: row_has_high_zeros(row, threshold), axis=1)]

        # Check even number of rows
        if data_in.shape[0] % 2 != 0:
            totMU = data_in.shape[0] - 1
        else:
            totMU = data_in.shape[0]

        # Check if there are MU's in the current data
        if totMU < 1:
            continue

        data = data_in.iloc[:totMU, :]

        num_MUs = int(data.shape[0])
        num_samp = int(data.shape[1])

        # Calculate the number of samples per trial
        samples_per_block = num_samp // n_trials

        # Reshape the data into trials
        trials = np.array_split(data.to_numpy(), n_trials, axis=1)

        trial_ranking = []

        input_dim = int(num_MUs)  # Number of MUs per pop

        for ind in range(n_trials):
            cur_trial = pd.DataFrame(trials[ind])

            ranking = []

            # Loop over k iterations
            for ite in range(num_iterations):

                # Get MU indices
                original_indices = list(cur_trial.index)

                # Shuffle the list of original indices
                shuffled_indices = np.random.permutation(original_indices)
                shuffled_samples = cur_trial.loc[shuffled_indices].T

                # Create a new instance of the autoencoder model
                m_1 = build_linear_vae(input_dim, latent_dim)

                # Train the models on the sampled matrices
                m_1.fit(shuffled_samples, shuffled_samples, epochs=epochs, batch_size=batch_size, verbose=0)

                # print(f"\n Calculating error for current iteration...")
                # Retrieve reconstructed signals
                recon = m_1.predict(shuffled_samples, verbose=0)

                # Calculate the variance explained by the reconstructed signal
                total_set = shuffled_samples.T

                r_squared_values = []
                # Calculate R-squared values for each pair of original and reconstructed time series
                for k in range(total_set.shape[0]):
                    original_series = np.asarray(total_set)[k, :]
                    reconstructed_series = recon.T[k, :]

                    ss_total = np.sum((original_series - np.mean(original_series)) ** 2)
                    ss_residual = np.sum((original_series - reconstructed_series) ** 2)
                    R2 = (shuffled_indices[k], 1 - (ss_residual / ss_total))
                    r_squared_values.append(R2)  # 1 - (ss_residual / ss_total))
                ranking.append(r_squared_values)

            # Create a defaultdict to accumulate scores for each name
            name_scores = defaultdict(list)

            # Iterate over each list and accumulate scores for each name
            for lst in ranking:
                for name, score in lst:
                    name_scores[name].append(score)

            # Create a new list of tuples containing name, score_mean, score_std
            result_list = []
            for name, scores in name_scores.items():
                mean_score = statistics.mean(scores)
                std_dev_score = statistics.stdev(scores) if len(scores) > 1 else 0  # To avoid division by zero
                result_list.append((name, mean_score, std_dev_score))

            # Sort the result list based on name
            result_list = sorted(result_list, key=lambda x: x[1], reverse=True)

            trial_ranking.append(result_list)
        rank_cur_con.append(trial_ranking)
    cv.append(rank_cur_con)

with open('l.npy', 'wb') as f:
    np.save(f, np.array(cv))

