import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import joblib
import re
import logging
import os
import torch.optim as optim
import json
import network
from scipy.stats import spearmanr
from scipy.signal import find_peaks, detrend
from dtaidistance import dtw
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date, timedelta, datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test():
    print('Test successful')

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def vae_loss_function2(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD + BCE * 0.1

def cyclic_time(df):
    df['tstp'] = pd.to_datetime(df['tstp'])
    
    # Create date and time columns directly from 'tstp'
    df['date_sin'] = np.sin(df['tstp'].dt.dayofyear * (2 * np.pi / 365))
    df['date_cos'] = np.cos(df['tstp'].dt.dayofyear * (2 * np.pi / 365))
    
    df['total_minutes'] = df['tstp'].dt.hour * 60 + df['tstp'].dt.minute
    df['time_sin'] = np.sin(df['total_minutes'] * (2 * np.pi / 1440))
    df['time_cos'] = np.cos(df['total_minutes'] * (2 * np.pi / 1440))

    df = df.drop(['tstp', 'total_minutes'], axis=1)
    
    df = df.round(4)
    return df

def get_medoid_loaders(batch_size):
    weather_data = torch.load('../../Data_processed/VAE/weather_data.pt')
    cluster_data = torch.load('../../Data_processed/VAE/cluster_data.pt')
    time_data = torch.load('../../Data_processed/VAE/time_data.pt')
    statistical_data = torch.load('../../Data_processed/VAE/statistical_data.pt')
    spike_data = torch.load('../../Data_processed/VAE/spike_data.pt')
    energy_data = torch.load('../../Data_processed/VAE/energy_data.pt')

    weather_data_tensors = torch.FloatTensor(weather_data)
    cluster_data_tensors = torch.FloatTensor(cluster_data)
    time_data_tensors = torch.FloatTensor(time_data)
    statistical_data_tensors = torch.FloatTensor(statistical_data)
    spike_data_tensors = torch.FloatTensor(spike_data)
    energy_data_tensors = torch.FloatTensor(energy_data)

    medoid_dataset = TensorDataset(weather_data_tensors, cluster_data_tensors, time_data_tensors, statistical_data_tensors, spike_data_tensors, energy_data_tensors)

    batch_size = batch_size
    medoid_loader = DataLoader(medoid_dataset, batch_size=batch_size, shuffle=True)

    return medoid_loader


def get_m2s_loader(batch_size):
    weather_data_m2s = torch.load('../../Data_processed/VAE/weather_data_m2s.pt')
    cluster_data_m2s = torch.load('../../Data_processed/VAE/cluster_data_m2s.pt')
    time_data_m2s = torch.load('../../Data_processed/VAE/time_data_m2s.pt')
    statistical_data_m2s = torch.load('../../Data_processed/VAE/statistical_data_m2s.pt')
    spike_data_m2s = torch.load('../../Data_processed/VAE/spike_data_m2s.pt')
    energy_data_m2s = torch.load('../../Data_processed/VAE/energy_data_m2s.pt')

    weather_data_r2s_tensors = torch.FloatTensor(weather_data_m2s)
    cluster_data_r2s_tensors = torch.FloatTensor(cluster_data_m2s)
    time_data_r2s_tensors = torch.FloatTensor(time_data_m2s)
    statistical_data_r2s_tensors = torch.FloatTensor(statistical_data_m2s)
    spike_data_r2s_tensors = torch.FloatTensor(spike_data_m2s)
    energy_data_r2s_tensors = torch.FloatTensor(energy_data_m2s)

    r2s_dataset = TensorDataset(weather_data_r2s_tensors, cluster_data_r2s_tensors, time_data_r2s_tensors, statistical_data_r2s_tensors, spike_data_r2s_tensors, energy_data_r2s_tensors)
    batch_size = batch_size
    r2s_loader = DataLoader(r2s_dataset, batch_size=batch_size, shuffle=True)

    return r2s_loader

def get_information(sequence_length, df):
    expected_length = sequence_length

    # Extract relevant columns
    weather_columns = ['temperature', 'windSpeed', 'humidity']
    cluster_columns = ['kmedoid_clusters']
    time_columns = ['date_sin', 'date_cos', 'time_sin', 'time_cos']
    statistical_columns = ['mean', 'median', 'std',  'min', 'max', 'gradient']
    spike_columns = ['spike_type', 'spike_magnitude']
    smoothed_energy_columns = ['energy(kWh/hh)_smoothed']
    energy_column = ['energy(kWh/hh)']

    weather = df[weather_columns].values
    cluster = df[cluster_columns].values
    time = df[time_columns].values
    statistical = df[statistical_columns].values
    spike = df[spike_columns].values
    smoothed_energy = df[smoothed_energy_columns].values
    real_energy = df[energy_column].values

    random_start = np.random.randint(0, len(real_energy) - expected_length)

    weather = weather[random_start:random_start + expected_length]
    cluster = cluster[random_start:random_start + expected_length]
    time = time[random_start:random_start + expected_length]
    statistical = statistical[random_start:random_start + expected_length]
    spike = spike[random_start:random_start + expected_length]
    smoothed_energy = smoothed_energy[random_start:random_start + expected_length]
    real_energy = real_energy[random_start:random_start + expected_length]

    weather = torch.FloatTensor(weather).to(device)
    cluster = torch.FloatTensor(cluster).to(device)
    time = torch.FloatTensor(time).to(device)
    statistical = torch.FloatTensor(statistical).to(device)
    spike = torch.FloatTensor(spike).to(device)

    weather = weather.unsqueeze(0)
    cluster = cluster.unsqueeze(0).int()
    time = time.unsqueeze(0)
    statistical = statistical.unsqueeze(0)
    spike = spike.unsqueeze(0)
    spike_type = spike[:, :, 0:1].int()
    spike_magnitude = spike[:, :, 1:]
    noise = torch.randn(1, expected_length, get_m_latent_dim()).to(device)

    return weather, cluster, time, statistical, spike_type, spike_magnitude, noise, real_energy

def weather_info(start, end):
    """
    Returns the weather dataframe for the given time period, with a 30 minute interval.
    Valid time period is from 2011-11-01 00:00:00 to 2014-0-21 22:00:00
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    weather_df = pd.read_csv("../../Data_raw/weather_hourly_darksky.csv")
    weather_df = weather_df.drop(['precipType', 'icon', 'summary', 'apparentTemperature', 'visibility', 'windBearing', 'dewPoint', 'pressure', 'apparentTemperature'], axis=1)
    weather_df['tstp'] = pd.to_datetime(weather_df['time'])
    weather_df = weather_df[(weather_df['tstp'] >= start) & (weather_df['tstp'] <= end)]
    weather_df.drop(['time'], axis=1, inplace=True)
    weather_df = weather_df.set_index('tstp')
    weather_df = weather_df.resample('30T').mean().interpolate()
    weather_df = weather_df.reset_index()

    # Min-max normalization
    for col in ['temperature', 'windSpeed', 'humidity']:
        min_col = weather_df[col].min()
        max_col = weather_df[col].max()
        weather_df[col] = (weather_df[col] - min_col) / (max_col - min_col)

    weather_df = weather_df.round(4)
    
    return weather_df

def get_spike_df_input(spike_hours, pre_spike_length, post_spike_length, spike_magnitudes):
    '''
    spike_hours: list of strings in the format 'HH:MM:SS'
    spike_durations: list of integers
    Should be the same length

    Returns a DataFrame with the time and the type of spike
    0: No spike
    1: Before the spike
    2: Spike Peak
    3: After the spike
    
    The DataFrame is for 48 hours in case of the spike being at the end of the day

    Looks like:
    time       spike_type
    00:00:00   0
    00:30:00   0
    01:00:00   0
    01:30:00   0
    02:00:00   1
    02:30:00   2
    03:00:00   3
    03:30:00   3
    04:00:00   0
    04:30:00   0

    '''
    spike_df = pd.DataFrame()
    spike_df['time'] = pd.date_range('00:00:00', '23:30:00', freq='30T')
    # Double it to make 48 hours
    # spike_df = pd.concat([spike_df, spike_df], ignore_index=True)
    spike_df['time'] = spike_df['time'].dt.time
    spike_df['spike_type'] = 0
    spike_df['spike_magnitude'] = 0.0

    spike_hours_time = [pd.to_datetime(time, format='%H:%M:%S').time() for time in spike_hours]
    spike_df.loc[spike_df['time'].isin(spike_hours_time), 'spike_type'] = 3
    spike_df.loc[spike_df['time'].isin(spike_hours_time), 'spike_magnitude'] = spike_magnitudes

    # Update for the row before the spike based on the pre_spike_length
    for spike_time, pre_spike_length in zip(spike_hours_time, pre_spike_length):
        spike_index = spike_df[spike_df['time'] == spike_time].index[0]
        for i in range(1, pre_spike_length + 1):
            if spike_index - i >= 0 and spike_df.loc[spike_index - i, 'spike_type'] == 0:
                spike_df.loc[spike_index - i, 'spike_type'] = 1

    # Update for the row after the spike based on the post_spike_length
    for spike_time, post_spike_length in zip(spike_hours_time, post_spike_length):
        spike_index = spike_df[spike_df['time'] == spike_time].index[0]
        for i in range(1, post_spike_length + 1):
            if spike_index + i < len(spike_df) and spike_df.loc[spike_index + i, 'spike_type'] == 0:
                spike_df.loc[spike_index + i, 'spike_type'] = 3
    # for index in spike_df[spike_df['spike_type'] == 3].index:
    #     if index > 0:
    #         spike_df.loc[index - 1, 'spike_type'] = 1

    # # Update for rows after the spike based on duration
    # for spike_time, duration in zip(spike_hours_time, spike_durations):
    #     spike_indices = spike_df[spike_df['time'] == spike_time].index

    #     for index in spike_indices:
    #         # No need to check if spike_df.loc[index + i, 'spike_type'] == 0
    #         for i in range(1, duration + 1):
    #             if index + i < len(spike_df):
    #                 spike_df.loc[index + i, 'spike_type'] = 4
    
    return spike_df

def merge_weather_spike(weather_df, spike_df):
    '''
    Merge the weather and spike DataFrames

    weather_df: DataFrame with the weather information
    spike_df: DataFrame with the spike information

    If the length of weather_df is less than spike_df, the spike_df is trimmed to match the length of weather_df
    If the length of weather_df is greater than spike_df, the spike_df is replicated to match the length of weather_df

    Returns a DataFrame with the weather and spike information merged
    
    '''

    if len(weather_df) < len(spike_df):
        trimmed_spike_df = spike_df[:len(weather_df)]
        merged_df = pd.concat([weather_df, trimmed_spike_df.reset_index(drop=True)], axis=1)
    else:
        replicate_times = len(weather_df) // len(spike_df) + (1 if len(weather_df) % len(spike_df) != 0 else 0)
        # Replicate spike_df
        replicated_spike_df = pd.concat([spike_df] * replicate_times, ignore_index=True)
        # Trim the replicated_spike_df to match the length of weather_df
        trimmed_replicated_spike_df = replicated_spike_df[:len(weather_df)]
        # Merge the DataFrames
        merged_df = pd.concat([weather_df.reset_index(drop=True), trimmed_replicated_spike_df.reset_index(drop=True)], axis=1)

    merged_df.drop(columns=['time'], inplace=True)
    return merged_df

def merge_statistics(w_spike_df, statistics):
    '''
    Merge the weather and spike DataFrame with the statistics

    w_spike_df: DataFrame with the weather and spike information
    statistics: list with the statistics in the following order: mean, median, std, min, max, gradient

    K-Medoids clustering is used to add the cluster information

    Returns a DataFrame with the weather, spike, and statistics information merged
    '''
    # Add noise to the statistics
    statistics = statistics + abs(np.random.normal(0, 0.01, len(statistics)))
    
    statistics_df = pd.DataFrame({'mean': statistics[0], 'median': statistics[1], 'std': statistics[2], 'min': statistics[3], 'max': statistics[4], 'gradient': statistics[5]}, index=[0])
        
    X = statistics_df.values
    kmoid_cluster = joblib.load('../Data_preprocess/kmedoids_model.joblib')
    cluster = kmoid_cluster.predict(X)
    statistics_df['kmedoid_clusters'] = cluster[0]

    kmoid_cluster = joblib.load('../Data_preprocess/kmedoids_model.joblib')
    cluster = kmoid_cluster.predict(X)
    statistics_df['kmedoid_clusters'] = cluster[0]

    
    statistics_df = pd.concat([statistics_df]*w_spike_df.shape[0], ignore_index=True)
    merged_df = pd.concat([w_spike_df, statistics_df], axis=1)
    return merged_df

def trim_and_merge_spike_weather(weather_df, spike_df):
    # Convert 'tstp' and 'time' to datetime if they are not already
    weather_df['tstp'] = pd.to_datetime(weather_df['tstp'])
    
    # Extract the date part from 'tstp' for merging
    weather_df['time'] = weather_df['tstp'].dt.time

    # Get the start and end time of the weather data
    weather_start_time = weather_df['time'].min()
    weather_end_time = weather_df['time'].max()

    # Convert 'time' in spike_df to datetime.time for comparison
    spike_df['time'] = pd.to_datetime(spike_df['time'], format='%H:%M:%S').dt.time

    # Trim the spike data to match the time range of the weather data
    trimmed_spike_df = spike_df[(spike_df['time'] >= weather_start_time) & (spike_df['time'] <= weather_end_time)]

    # Merge the weather data and the trimmed spike data
    merged_df = pd.merge(weather_df, trimmed_spike_df, on='time', how='left')

    return merged_df

def enhanced_energy_profile(spike_type, synthetic_energy, gradient, min_energy):
    '''
    Function to enhance the energy profile by adding periodic noises

    Args:
    spike_type (Tensor): Spike type tensor
    synthetic_energy (Tensor): Synthetic energy tensor
    gradient (float): Gradient value
    min_energy (float): Minimum energy value

    Returns:
    fake_energy_enhanced (Tensor): Enhanced energy profile
    '''
    selected_period = get_periodic()
    frequency = 2 * np.pi / selected_period
    phase_shift = np.random.uniform(0, 2 * np.pi)

    fake_energy_enhanced = synthetic_energy.copy()

    zero_run_length = 0
    counter = 0
    random_p = [np.random.normal(0, 1/5 * gradient) for _ in range(selected_period)]
    s_type = spike_type.squeeze().detach().cpu().numpy()

    mean_e = np.mean(synthetic_energy)

    for i in range(len(s_type)):
        if s_type[i] == 0 and synthetic_energy[i] < mean_e:
            zero_run_length += 1
            # if zero_run_length == 5:
            #     # Start enhancing from i-4 to i
            #     for j in range(i - 4, i + 1):
            #         fake_energy_enhanced[j] = max(min_energy, synthetic_energy[j] + random_p[counter % selected_period])
            #         counter += 1
            # elif zero_run_length > 5:
            #     # Continue enhancing
            #     fake_energy_enhanced[i] = max(min_energy, synthetic_energy[i] + random_p[counter % selected_period])
            #     counter += 1
            if zero_run_length >= 5:
                # Add sinusoidal noise
                i_tensor = torch.tensor(i, dtype=torch.float32)
                sinusoidal_noise = 1/5 * gradient * torch.sin(frequency * i_tensor + phase_shift)  # Sine wave noise
                fake_energy_enhanced[i] = max(min_energy, synthetic_energy[i] + sinusoidal_noise)
        else:
            zero_run_length = 0
            # counter = 0

    return fake_energy_enhanced

def generate_synthetic_energy(start_date_time = None, end_date_time = None, statistics = None, spike_hours = None, pre_spike_length = None, post_spike_length = None, model_medoid = None, model_m2s = None, device = None):
    '''
    Generate synthetic energy data

    start_date_time: string with the start date and time in the format 'YYYY-MM-DD HH:MM:SS'
    end_date_time: string with the end date and time in the format 'YYYY-MM-DD HH:MM:SS'

    statistics: list with the statistics in the following order: mean, median, std, min, max, gradient

    spike_hours: list of strings in the format 'HH:MM:SS'
    spike_durations: list of integers
    Should be the same length

    Returns a DataFrame with the time, and the synthetic energy data

    time_df
    tstp       
    2013-06-25 00:00:00   
    2013-06-25 00:30:00
    2013-06-25 01:00:00
    ...

    synthetic_energy
    [0.1, 0.2, 0.15, ...]

    '''
    weather_df = weather_info(start_date_time, end_date_time)
    time_df = weather_df[['tstp']]

    if spike_hours and pre_spike_length and post_spike_length:
        spike_magnitude = statistics[4]
        spike_df = get_spike_df_input(spike_hours, pre_spike_length, post_spike_length, spike_magnitude)
        w_spike_df = merge_weather_spike(weather_df, spike_df)
    else:
        mean = statistics[0]
        std  = statistics[2]
        max = statistics[4]
        statistics_df = pd.DataFrame({'mean': statistics[0], 'median': statistics[1], 'std': statistics[2], 'min': statistics[3], 'max': statistics[4], 'gradient': statistics[5]}, index=[0])
        X = statistics_df.values
        kmoid_cluster = joblib.load('../Data_preprocess/kmedoids_model.joblib')
        cluster = kmoid_cluster.predict(X)[0]
        month = int(start_date_time.split('-')[1])

        spike_df = get_spike_df(cluster, month, mean, std, max)
        start_date_str = start_date_time.split()[0]
        end_date_str = end_date_time.split()[0]

        # Convert the date strings to datetime objects
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        # Calculate the difference in days
        num_days = (end_date - start_date).days + 1  # Including both start and end dates

        spike_df_days = []
        
        for i in range(num_days):
            spike_df = get_spike_df(cluster, month, mean, std, max)
            spike_df_days.append(spike_df)

        spike_df_days = pd.concat(spike_df_days, ignore_index=True)
        w_spike_df = trim_and_merge_spike_weather(weather_df, spike_df_days)
        
    w_spike_df = cyclic_time(w_spike_df)

    w_spike_s_df = merge_statistics(w_spike_df, statistics)

    noise = torch.randn(1, len(w_spike_s_df), get_m_latent_dim()).to(device)

    weather_columns = ['temperature', 'windSpeed', 'humidity']
    cluster_columns = ['kmedoid_clusters']
    time_columns = ['date_sin', 'date_cos', 'time_sin', 'time_cos']
    statistical_columns = ['mean', 'median', 'std',  'min', 'max', 'gradient']
    spike_columns = ['spike_type', 'spike_magnitude']

    weather = torch.tensor(w_spike_s_df[weather_columns].values, dtype=torch.float32).unsqueeze(0).to(device)
    cluster = torch.tensor(w_spike_s_df[cluster_columns].values, dtype=torch.float32).unsqueeze(0).to(device).int()
    time = torch.tensor(w_spike_s_df[time_columns].values, dtype=torch.float32).unsqueeze(0).to(device)
    statistical = torch.tensor(w_spike_s_df[statistical_columns].values, dtype=torch.float32).unsqueeze(0).to(device)
    spike = torch.tensor(w_spike_s_df[spike_columns].values, dtype=torch.float32).unsqueeze(0).to(device).int()

    spike_type = spike[:, :, 0:1].int().to(device)
    spike_magnitude = spike[:, :, 1:].to(device)

    h = model_medoid.lstm_decoder(noise, weather, cluster, time)
    # Add noise to the hidden state
    h = h + torch.randn(h.shape).to(device) * 0.1

    latent_space = model_m2s.lstm_encoder(h)
    mu, logvar = latent_space.chunk(2, dim=2)
    z = model_m2s.reparameterize(mu, logvar)
    h = model_m2s.lstm_decoder(z, weather, cluster, time, statistical, spike_type, spike_magnitude)

    synthetic_energy = h.squeeze().detach().cpu().numpy()

    min_energy = statistical.squeeze().detach().cpu().numpy()[0][3]
    gradient = statistical.squeeze().detach().cpu().numpy()[0][-1]
    enhanced_synthetic_energy = enhanced_energy_profile(spike_type, synthetic_energy, gradient, min_energy)

    return time_df, synthetic_energy, enhanced_synthetic_energy

def add_missing_pre_post_spikes(spike_df):
    for index, row in spike_df.iterrows():
        if row['spike_type'] == 2 or row['spike_type'] == 3:
            # Check for pre spikes
            if index > 0 and spike_df.at[index - 1, 'spike_type'] == 0:
                # Add 2 pre spikes if the previous type is 0
                for _ in range(2):
                    pre_time = (pd.to_datetime(row['time'].strftime('%H:%M:%S')) - pd.Timedelta(minutes=30)).time()
                    spike_df.loc[spike_df['time'] == pre_time, 'spike_type'] = 1  # Assign pre spike
            # Check for post spikes
            if index < len(spike_df) - 1 and spike_df.at[index + 1, 'spike_type'] == 0:
                # Add 2 post spikes if the next type is 0
                for _ in range(2):
                    post_time = (pd.to_datetime(row['time'].strftime('%H:%M:%S')) + pd.Timedelta(minutes=30)).time()
                    spike_df.loc[spike_df['time'] == post_time, 'spike_type'] = 4  # Assign post spike

    return spike_df


def get_time_spike_prob():

    average_prob_time = pd.read_csv('../../Data_processed/average_prob_time.csv')
    average_prob_time = average_prob_time[average_prob_time['time'] != '12:54']
    average_prob_time = average_prob_time[average_prob_time['time'] != '13:15']
    average_prob_time = average_prob_time.reset_index(drop=True)
    probabilities_spike_time = average_prob_time['average_spike_probability'] / average_prob_time['average_spike_probability'].sum()

    return average_prob_time, probabilities_spike_time

def add_pre_post_spikes(spike_df, prespike_before_2, prespike_before_3, postspike_after_2, postspike_after_3):
    for i in range(len(spike_df)):
        if spike_df.loc[i, 'spike_type'] == 2:
            if i - prespike_before_2 < 0:
                spike_df.loc[0:i - 1, 'spike_type'] = 1
            else:
                spike_df.loc[i - prespike_before_2:i - 1, 'spike_type'] = 1

            if i + postspike_after_2 >= len(spike_df):
                spike_df.loc[i + 1:len(spike_df), 'spike_type'] = 4
            else:
                spike_df.loc[i + 1:i + postspike_after_2, 'spike_type'] = 4
        elif spike_df.loc[i, 'spike_type'] == 3:
            if i - prespike_before_3 < 0:
                spike_df.loc[0:i - 1, 'spike_type'] = 1
            else:
                spike_df.loc[i - prespike_before_3:i - 1, 'spike_type'] = 1
            if i + postspike_after_3 >= len(spike_df):
                spike_df.loc[i + 1:len(spike_df), 'spike_type'] = 4
            else:
                spike_df.loc[i + 1:i + postspike_after_3, 'spike_type'] = 4
    return spike_df

def get_random_spiked_day():

    # Mean and standard deviation for the number of prespikes and postspikes for each spike type
    means = [1.53, 1.87, 1.74, 2.24]
    stds = [1.17, 1.76, 1.56, 2.18]
    # Mean and standard deviation for the number of spikes per day
    mean_num_spikes = 1.92
    std_num_spikes = 0.42
    # Probabilities for the spike types
    probabilities_spike_type = [0.53, 0.47]
    values = [3, 2]

    # Generate the number of prespikes and postspikes for each spike type
    random_pre_post_numbers = [int(np.random.normal(mean, std)) for mean, std in zip(means, stds)]
    random_pre_post_numbers = [max(1, num) for num in random_pre_post_numbers]
    random_pre_post_numbers = [min(5, num) for num in random_pre_post_numbers]
    # Assign the values to the variables
    prespike_before_2 = random_pre_post_numbers[0]
    prespike_before_3 = random_pre_post_numbers[1]
    postspike_after_2 = random_pre_post_numbers[2]
    postspike_after_3 = random_pre_post_numbers[3]
    
    # Generate the number of spikes per day
    num_spikes_per_day = round(np.random.normal(mean_num_spikes, std_num_spikes))
    num_spikes_per_day = max(1, num_spikes_per_day)
    num_spikes_per_day = min(5, num_spikes_per_day)
    
    # Generate the array
    generated_spike_type_array = np.random.choice(values, size=num_spikes_per_day, p=probabilities_spike_type)

    # Get the time and spike probability
    average_prob_time, probabilities_spike_time = get_time_spike_prob()
    
    flag = True
    while flag:
        selected_indices = np.random.choice(average_prob_time.index, size=num_spikes_per_day, p=probabilities_spike_time, replace=False)
        selected_indices = np.sort(selected_indices)

        required_spaces_before = [prespike_before_2 if spike == 2 else prespike_before_3 for spike in generated_spike_type_array]
        required_spaces_after = [postspike_after_2 if spike == 2 else postspike_after_3 for spike in generated_spike_type_array]
        can_fit_spikes = all(selected_indices[i] - required_spaces_before[i] > selected_indices[i-1] + required_spaces_after[i-1]
                            for i in range(1, len(selected_indices)))
        
        if can_fit_spikes:
            flag = False

    # Get the selected rows
    selected_rows = average_prob_time.iloc[selected_indices]
    spike = pd.DataFrame()
    spike['tstp'] = average_prob_time['time']
    selected_time = selected_rows['time'].tolist()

    spike['spike_type'] = 0

    spike.loc[spike['tstp'].isin(selected_time), 'spike_type'] = generated_spike_type_array

    spike = add_pre_post_spikes(spike, prespike_before_2, prespike_before_3, postspike_after_2, postspike_after_3)
    return spike

# class QualityJudge(nn.Module):
#     '''
#     Quality Judge model

#     Simple feedforward neural network with 2 hidden layers.
#     Use [input] data to predict [output] data.

#     input_size: Number of features in the input
#     output_size: Number of features in the output
#     '''
#     def __init__(self, input_size, output_size):
#         super(QualityJudge, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_size),
#             nn.ReLU()
#         )
    
#     def forward(self, x):
#         return self.model(x)

# def trtr_error(X_real, y_real, X_real2, y_real2, criterion):
#     '''
#     Train on real, test on real.

#     Train the model on real data: X_real, y_real
#     Test the model on real data, use X_real to generate y_real_pred
#     Compare y_real_pred with y_real using criterion

#     X_real: Real data
#     y_real: Real labels
#     '''
#     input_size = X_real.shape[1]
#     output_size = y_real.shape[1]
#     quality_judge = QualityJudge(input_size, output_size)
#     optimizer = torch.optim.Adam(quality_judge.parameters(), lr=0.01)
#     # Train on real, test on real
#     for epoch in range(5):
#         optimizer.zero_grad()
#         output = quality_judge(X_real)
#         loss = criterion(output, y_real)
#         loss.backward()
#         optimizer.step()

#     test_on_real = quality_judge(X_real2)
#     return criterion(test_on_real, y_real2).item()

# def trts_error(X_real, y_real, X_fake, y_fake, criterion):
#     '''
#     Train on real, test on fake.
    
#     Train the model on real data: X_real, y_real
#     Test the model on fake data, use X_fake to generate y_fake_pred
#     Compare y_fake_pred with y_fake using criterion

#     X_real: Real data
#     y_real: Real labels
#     X_fake: Fake data
#     y_fake: Fake labels
#     '''
#     input_size = X_real.shape[1]
#     output_size = y_real.shape[1]
#     quality_judge = QualityJudge(input_size, output_size)
#     optimizer = torch.optim.Adam(quality_judge.parameters(), lr=0.01)
#     # Train on real, test on fake
#     for epoch in range(5):
#         optimizer.zero_grad()
#         output = quality_judge(X_real)
#         loss = criterion(output, y_real)
#         loss.backward()
#         optimizer.step()

#     test_on_fake = quality_judge(X_fake)
#     return criterion(test_on_fake, y_fake).item()

# def tstr_error(X_fake, y_fake, X_real, y_real, criterion):
#     '''
#     Train on fake, test on real.

#     Train the model on fake data: X_fake, y_fake
#     Test the model on real data, use X_real to generate y_real_pred
#     Compare y_real_pred with y_real using criterion

#     X_fake: Fake data
#     y_fake: Fake labels
#     X_real: Real data
#     y_real: Real labels
#     '''
#     input_size = X_real.shape[1]
#     output_size = y_real.shape[1]
#     quality_judge = QualityJudge(input_size, output_size)
#     optimizer = torch.optim.Adam(quality_judge.parameters(), lr=0.01)
#     # Train on fake, test on real
#     for epoch in range(5):
#         optimizer.zero_grad()
#         output = quality_judge(X_fake)
#         loss = criterion(output, y_fake)
#         loss.backward()
#         optimizer.step()

#     test_on_real = quality_judge(X_real)
#     return criterion(test_on_real, y_real).item()

# def tsts_error(X_fake, y_fake, X_fake2, y_fake2, criterion):
#     '''
#     Train on fake, test on fake.

#     Train the model on fake data: X_fake, y_fake
#     Test the model on fake data, use X_fake to generate y_fake_pred
#     Compare y_fake_pred with y_fake using criterion
    
#     X_fake: Fake data
#     y_fake: Fake labels
#     '''
#     input_size = X_fake.shape[1]
#     output_size = y_fake.shape[1]
#     quality_judge = QualityJudge(input_size, output_size)
#     optimizer = torch.optim.Adam(quality_judge.parameters(), lr=0.01)
#     # Train on fake, test on fake
#     for epoch in range(5):
#         optimizer.zero_grad()
#         output = quality_judge(X_fake)
#         loss = criterion(output, y_fake)
#         loss.backward()
#         optimizer.step()

#     test_on_fake = quality_judge(X_fake2)
#     return criterion(test_on_fake, y_fake2).item()

def generate_random_spike_count(cluster, month):
    spike_num_prob_df = pd.read_csv('../../Data_processed/Spike_count_cluster_month.csv')
    temp_df = spike_num_prob_df[spike_num_prob_df['kmedoid_clusters'] == cluster]
    selected_spike_count = np.random.choice(temp_df['spike_count'], p=temp_df[f'month_{month}'])
    return int(selected_spike_count)

def parse_and_format_time(time_quad_str):
    # Extract all time occurrences using regex
    times = re.findall(r'\d+, \d+', time_quad_str)
    # Convert each found time into a formatted string
    formatted_times = [f"{int(hour):02d}:{int(minute):02d}:00" for hour, minute in (time.split(', ') for time in times)]
    return tuple(formatted_times)

def get_spike_time(cluster, spike_count):
    spike_count = int(spike_count)
    spike_time_prob_df = pd.read_csv(f'../../Data_processed/statistics/{spike_count}spike_time_pair.csv')
    
    if spike_count == 1:
        pass
    else:
        spike_time_prob_df['time_pairs'] = spike_time_prob_df['time_pairs'].apply(parse_and_format_time)

    spike_time_prob_df = spike_time_prob_df[spike_time_prob_df['clusters'] == cluster]
    
    spike_time_prob_df['probability'] /= spike_time_prob_df['probability'].sum()

    selected_time = np.random.choice(spike_time_prob_df['time_pairs'], p=spike_time_prob_df['probability'])

    return selected_time

# def get_spike_type(cluster, spike_count):
#     spike_count = int(spike_count)
#     spike_type_prob_df = pd.read_csv(f'../../Data_processed/statistics/{spike_count}spike_type_prob.csv')
#     spike_type_prob_df = spike_type_prob_df[spike_type_prob_df['clusters'] == cluster]
#     spike_type_prob_df['probability'] /= spike_type_prob_df['probability'].sum()

#     selected_spike_type = []
#     for i in range(spike_count):
#         selected_spike_type.append(np.random.choice(spike_type_prob_df['spike_type'], p=spike_type_prob_df['probability']))
    
#     return selected_spike_type
def get_spike_type(cluster, spike_count):
    spike_count = int(spike_count)
    spike_type_prob_df = pd.read_csv(f'../../Data_processed/statistics/1spike_type_prob.csv')
    spike_type_prob_df = spike_type_prob_df[spike_type_prob_df['clusters'] == cluster]
    spike_type_prob_df['probability'] /= spike_type_prob_df['probability'].sum()

    selected_spike_type = []
    for i in range(spike_count):
        selected_spike_type.append(np.random.choice(spike_type_prob_df['spike_type'], p=spike_type_prob_df['probability']))
    
    return selected_spike_type

# def get_pre_post_spike(cluster, spike_count):
#     spike_count = int(spike_count)
#     cluster_df_12 = pd.read_csv(f'../../Data_processed/statistics/{spike_count}spike_12_clusters.csv')
#     cluster_df_13 = pd.read_csv(f'../../Data_processed/statistics/{spike_count}spike_13_clusters.csv')
#     cluster_df_24 = pd.read_csv(f'../../Data_processed/statistics/{spike_count}spike_24_clusters.csv')
#     cluster_df_34 = pd.read_csv(f'../../Data_processed/statistics/{spike_count}spike_34_clusters.csv')
#     cluster_df_12 = cluster_df_12[cluster_df_12['cluster'] == cluster]
#     cluster_df_13 = cluster_df_13[cluster_df_13['cluster'] == cluster]
#     cluster_df_24 = cluster_df_24[cluster_df_24['cluster'] == cluster]
#     cluster_df_34 = cluster_df_34[cluster_df_34['cluster'] == cluster]
#     spike_12 = np.random.choice(cluster_df_12['count_1s_before_2'], p=cluster_df_12['probability'])
#     spike_13 = np.random.choice(cluster_df_13['count_1s_before_3'], p=cluster_df_13['probability'])
#     spike_24 = np.random.choice(cluster_df_24['count_4s_after_2'], p=cluster_df_24['probability'])
#     spike_34 = np.random.choice(cluster_df_34['count_4s_after_3'], p=cluster_df_34['probability'])
#     return spike_12, spike_13, spike_24, spike_34
def get_pre_post_spike(spike_type):
    # Load your DataFrame
    pre_post_df = pd.read_csv('../../Data_processed/Spike_pre_post_ones_fours.csv')
    
    # Filter for the specified spike_type
    temp_df = pre_post_df[pre_post_df['spike_type'] == spike_type]
    
    # Ensure that the DataFrame is not empty
    if temp_df.empty:
        print(f"No data found for spike_type {spike_type}.")
        return None
    
    # Select pre and post pairs based on their probabilities
    choices = temp_df[['pre_spike_1_count', 'post_spike_4_count']].values
    probabilities = temp_df['probability'].values
    
    # Randomly select a pair based on the probability distribution
    selected_pair = choices[np.random.choice(len(choices), p=probabilities / probabilities.sum())]
    
    return selected_pair

def get_spike_magnitude(spike_type, cluster, mean, std, max):
    count = len(spike_type)
    spike_mag_stats_df = pd.read_csv('../../Data_processed/statistics/spike_mag_stats.csv')
    spike_mag_stats_df = spike_mag_stats_df[spike_mag_stats_df['clusters'] == cluster]

    spike_mag = []

    for type in spike_type:

        if type == 2:
            # min_mag = mean + 2*std
            # max_mag = mean + 3*std
            min_mag = 0.5 * std + mean
            max_mag = 2.5 * std + mean
        else:
            min_mag = 2.5 * std + mean 
            max_mag = max
        
        temp_df = spike_mag_stats_df[spike_mag_stats_df['spike_type'] == type]
        mag = np.random.normal(temp_df['mean'], temp_df['std'])

        while mag < min_mag or mag > max_mag:
            mag = np.random.normal(temp_df['mean'], temp_df['std'])
        spike_mag.append(mag[0])

    return spike_mag

def get_spike_df(cluster, month, mean, std, max):
    spike_df = pd.DataFrame()
    spike_df['time'] = pd.date_range('00:00:00', '23:30:00', freq='30T').time
    spike_df['spike_type'] = 0
    spike_df['spike_magnitude'] = 1.0

    spike_count = generate_random_spike_count(cluster, month)

    if spike_count == 0:
        return spike_df  # Return the empty DataFrame if no spikes

    spike_type = []
    spike_time = []
    pre_post_spikes = []

    for i in range(spike_count):
        t_spike_type = generate_spike_type(cluster, month)
        spike_type.append(t_spike_type)

        t_spike_time = generate_spike_time(cluster, month, t_spike_type)
        # spike_time.append(t_spike_time)

        patience = 0

        while True:
            t_spike_time = generate_spike_time(cluster, month, t_spike_type)
            # t_spike_time_dt = pd.to_datetime(t_spike_time).time()

            # Check if the new spike time is at least 2 time steps away
            if all(abs((pd.to_datetime(t).hour * 60 + pd.to_datetime(t).minute) - 
                        (pd.to_datetime(t_spike_time).hour * 60 + pd.to_datetime(t_spike_time).minute)) 
                        >= 120 for t in spike_time):
                spike_time.append(t_spike_time)
                break  # Exit the while loop once a valid time is found

            patience += 1
            if patience > 100:
                print("Patience for arranging spike times exceeded, use the last valid time.")
                spike_time.append(t_spike_time)
                break
        
        pre_post_spikes.append(get_pre_post_spike(t_spike_type))

    # Convert spike_time to time format for comparison
    spike_time = [pd.to_datetime(t).time() for t in spike_time]

    # Calculate magnitude using the entire spike_type sequence
    magnitude = get_spike_magnitude(spike_type, cluster, mean, std, max)

    for t, st, mg, (pre_spike, post_spike) in zip(spike_time, spike_type, magnitude, pre_post_spikes):
        spike_df.loc[spike_df['time'] == t, 'spike_type'] = st
        spike_df.loc[spike_df['time'] == t, 'spike_magnitude'] = mg

        # Convert time to datetime for calculations
        t_dt = pd.to_datetime(str(t), format='%H:%M:%S')

        # Assign pre_spike values (1s)
        for _ in range(pre_spike):
            pre_time = (t_dt - pd.Timedelta(minutes=30)).time()  # 1 time step before
            if spike_df.loc[spike_df['time'] == pre_time, 'spike_type'].values[0] not in [2, 3]:
                spike_df.loc[spike_df['time'] == pre_time, 'spike_type'] = 1  # Assign 1
            t_dt = pd.to_datetime(pre_time, format='%H:%M:%S')  # Update t_dt for the next pre spike

        # Assign post_spike values (4s)
        t_dt = pd.to_datetime(str(t), format='%H:%M:%S')  # Reset t_dt for post spikes
        for _ in range(post_spike):
            post_time = (t_dt + pd.Timedelta(minutes=30)).time()  # 1 time step after
            if post_time == pd.Timestamp('00:00:00').time():
                break  # Prevent wrapping to the beginning of the day
            if spike_df.loc[spike_df['time'] == post_time, 'spike_type'].values[0] not in [2, 3]:
                spike_df.loc[spike_df['time'] == post_time, 'spike_type'] = 4  # Assign 4
            t_dt = pd.to_datetime(post_time, format='%H:%M:%S')  # Update t_dt for the next post spike

    return spike_df

def gradient_loss(synthetic_profile, statistical):

    # Calculate the gradient of the synthetic profile
    synthetic_gradient = torch.mean(abs(synthetic_profile[:, :-1, :] - synthetic_profile[:, 1:, :]), dim=1)

    # Get the target gradient from the last column of the statistical data
    target_gradient = statistical[:, 0, -1:]

    # Calculate the loss as the mean squared error between the gradients
    loss = torch.sum(abs(synthetic_gradient - target_gradient))
    
    return loss

def mVAE_hsearch(model_class, space, dataloader, device, model_name, num_epochs):
    best_loss = float('inf')
    best_config = {key: space[key][0] for key in space}  # Initialize with the first value of each hyperparameter

    for key in space:
        logging.info(f"Optimizing hyperparameter: {key}")
        
        for value in space[key]:
            current_config = best_config.copy()
            current_config[key] = value

            logging.info(f"Testing configuration: {current_config}")

            model = model_class(
                encoder_hidden=current_config['encoder_hidden'],
                encoder_lstm_size=current_config['encoder_lstm_size'],
                encoder_lstm_layers=current_config['encoder_lstm_layers'],
                latent_dim=current_config['latent_dim'],
                decoder_hidden_size=current_config['decoder_hidden_size'],
                embed_size=current_config['embed_size'],
                time_size=current_config['time_size'],
                decoder_lstm_size=current_config['decoder_lstm_size'],
                decoder_lstm_layers=current_config['decoder_lstm_layers']
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=5e-4)

            current_loss = m_train_for_search(model, num_epochs, dataloader, optimizer, device)
        
            if current_loss < best_loss:
                best_loss = current_loss
                best_config[key] = value

        logging.info(f'Best value for {key}: {best_config[key]}')
        logging.info(f'Current best configuration: {best_config}')
    
    logging.info(f'Best configuration: {best_config}')
    logging.info(f'Best loss: {best_loss}')

    return best_config, best_loss

def m_train_for_search(model, num_epochs, dataloader, optimizer, device):
    best_loss = float('inf')
    model.train()
    patience = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for _, (weather, cluster, time, _, _, real_energy) in enumerate(dataloader):
            weather = weather.to(device)
            cluster = cluster.to(device).int()
            time = time.to(device)
            real_energy = real_energy.to(device)

            optimizer.zero_grad()
            output, mu, logvar = model(weather, cluster, time, real_energy)
            loss = vae_loss_function(output, real_energy, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
        
        if patience == 5:
            break

    return best_loss

def get_m_latent_dim():
    with open('Config/medoid_config.json', 'r') as file:
        config = json.load(file)
    
    dim = config['latent_dim']
    return dim


def m2sVAE_hsearch(model_class, space, dataloader, device, model_name, num_epochs):
    best_loss = float('inf')
    best_config = {}

    # Initialize the best_config with the first values from the space
    for key in space:
        best_config[key] = space[key][0]

    for key in space:
        logging.info(f"Optimizing hyperparameter: {key}")

        best_value = None
        for value in space[key]:
            current_config = best_config.copy()
            current_config[key] = value

            # Load the medoid configuration
            with open('medoid_config.json', 'r') as file:
                medoid_config = json.load(file)
            model_medoid = network.m_VAE(**medoid_config).to(device)

            logging.info(f"Testing configuration: {current_config}")

            model_m2s = model_class(
                encoder_hidden=current_config['encoder_hidden'],
                encoder_lstm_size=current_config['encoder_lstm_size'],
                encoder_lstm_layers=current_config['encoder_lstm_layers'],
                latent_dim=current_config['latent_dim'],
                decoder_hidden=current_config['decoder_hidden'],
                embed_size_cluster=current_config['embed_size_cluster'],
                embed_size_spike=current_config['embed_size_spike'],
                decoder_layers=current_config['decoder_layers'],
                decoder_lstm_size=current_config['decoder_lstm_size'],
                decoder_lstm_dropout=current_config['decoder_lstm_dropout'],
                time_size=current_config['time_size'],
                statistics_size=current_config['statistics_size'],
                gradient_size=current_config['gradient_size'],
                fc_dropout=current_config['fc_dropout']
            ).to(device)

            optimizer = optim.Adam(model_m2s.parameters(), lr=3e-4)
        
            # Training process
            current_loss = m2s_train_for_search(model_medoid, model_m2s, num_epochs, dataloader, optimizer, device)
        
            if current_loss < best_loss:
                best_loss = current_loss
                best_value = value
                best_config[key] = value

        logging.info(f'Best value for {key}: {best_value}')
        logging.info(f'Current best configuration: {best_config}')
    
    logging.info(f'Best configuration: {best_config}')
    logging.info(f'Best loss: {best_loss}')

    return best_config, best_loss

def m2s_train_for_search(model_medoid, model_m2s, num_epochs, dataloader, optimizer, device):
    best_loss = float('inf')

    model_m2s.train()
    model_medoid.eval()

    patience = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for idx, (weather, cluster, time, statistical, spike, real_energy) in enumerate(dataloader):
            weather = weather.to(device)
            cluster = cluster.to(device).int()
            time = time.to(device)
            statistical = statistical.to(device)
            spike_type = spike[:, :, 0:1].to(dtype=torch.int32)
            spike_magnitude = spike[:, :, 1:].to(dtype=torch.float32)
            spike_type = spike_type.to(device)
            spike_magnitude = spike_magnitude.to(device)
            real_energy = real_energy.to(device)

            with torch.no_grad():
                noise = torch.randn(real_energy.shape[0], real_energy.shape[1], get_m_latent_dim()).to(device)
                medoid_synthetic = model_medoid.lstm_decoder(noise, weather, cluster, time)

            optimizer.zero_grad()
            synthetic_profile, mu, logvar = model_m2s(medoid_synthetic, weather, cluster, time, statistical, spike_type, spike_magnitude)

            loss = vae_loss_function(synthetic_profile, real_energy, mu, logvar)
            loss += 10 * gradient_loss(synthetic_profile, statistical)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
        
        if patience == 3:
            break

    return best_loss

def spearman_correlation(seq1, seq2):
    return spearmanr(seq1, seq2).correlation

def pearson_correlation(seq1, seq2):
    return np.corrcoef(seq1, seq2)[0, 1]

def mean_squared_error(seq1, seq2):
    return np.mean((seq1 - seq2) ** 2)

def dtw_distance(seq1, seq2):
    return dtw.distance(seq1, seq2)

def cosine_similarity_score(seq1, seq2):
    seq1 = np.array(seq1).reshape(1, -1)
    seq2 = np.array(seq2).reshape(1, -1)
    return cosine_similarity(seq1, seq2)[0, 0]

def get_periodic():
    '''
    Function to get the periodicity of the synthetic data

    Period,Probability
    3,     0.6628131021194612
    4,     0.21451509312780995
    2,     0.05651894669235695
    5,     0.03660886319845858
    6,     0.012845215157353623
    7,     0.00642260757867693
    8,     0.005780346820809225
    12,    0.0019267822736028786
    9,     0.0012845215157352916
    10,    0.0012845215157352916

    Returns:
    selected_period (int): Period of the synthetic data
    '''
    period_probability = pd.read_csv('../../Data_processed/period_prob_df.csv')
    df_filtered = period_probability[period_probability['Period'] == period_probability['Period'].astype(int)].copy()
    # Recalculating probabilities
    total_probability = df_filtered['Probability'].sum()
    df_filtered.loc[:, 'Probability'] = df_filtered['Probability'] / total_probability
    selected_period = np.random.choice(df_filtered['Period'], p=df_filtered['Probability'])
    selected_period = int(selected_period)
    return selected_period

# def enhanced_energy_profile(spike_type, synthetic_energy, gradient, min_energy):
#     '''
#     Function to enhance the energy profile by adding periodic noises

#     Args:
#     spike_type (Tensor): Spike type tensor
#     synthetic_energy (Tensor): Synthetic energy tensor
#     gradient (float): Gradient value

#     Returns:
#     fake_energy_enhanced (Tensor): Enhanced energy profile

#     '''
#     selected_period = utils.get_periodic()
#     fake_energy_enhanced = synthetic_energy.copy()

#     counter = 0
#     random_p = [np.random.normal(0, 1/3 * gradient) for _ in range(selected_period)]
#     # print(spike_type.shape)
#     s_type = spike_type.squeeze().detach().cpu().numpy()
#     # print(s_type)

#     for i in range(len(s_type)):
#         if s_type[i] == 0:
#             fake_energy_enhanced[i] = max(min_energy, synthetic_energy[i] + random_p[counter % selected_period])
#             counter += 1
#         else:
#             counter = 0

#     return fake_energy_enhanced

def find_period(data):
    if len(data) == 0:
        return None
    
    # Detrend the data to remove any linear trend
    detrended_data = detrend(data)

    # Autocorrelation to find repeating patterns
    autocorr = np.correlate(detrended_data, detrended_data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Find peaks in the autocorrelation to determine the period
    peaks, _ = find_peaks(autocorr, height=0)
    periods = np.diff(peaks)  # Differences between consecutive peaks

    # Estimate the period
    if len(periods) > 0:
        estimated_period = np.median(periods)
    else:
        estimated_period = None

    return estimated_period

def trim_and_merge_spike_weather(weather_df, spike_df):
    # Convert 'tstp' to datetime
    weather_df['tstp'] = pd.to_datetime(weather_df['tstp'])
    
    # Extract the time for merging
    weather_df['time'] = weather_df['tstp'].dt.time

    first_time = weather_df['time'].iloc[0]
    last_time = weather_df['time'].iloc[-1]

    # Ensure spike_df time is in datetime.time format
    spike_df['time'] = pd.to_datetime(spike_df['time'], format='%H:%M:%S').dt.time 
    spike_df.to_csv('spike_df.csv', index=False)

    # Find the time in the spike data that matches the first and last time in the weather data
    i = 0
    while spike_df['time'].iloc[i] < first_time:
        i += 1
    start_index = i

    i = 1
    while spike_df['time'].iloc[-i] > last_time:
        i += 1
    end_index = i - 1

    # Trim the spike data to match the time range of the weather data
    trimmed_spike_df = spike_df.iloc[start_index:-end_index]
    trimmed_spike_df.to_csv('trimmed_spike_df.csv', index=False)
    weather_df.to_csv('weather_df.csv', index=False)
    # print(trimmed_spike_df.shape)
    # print(weather_df.shape)
    merged_df = weather_df.copy()
    merged_df['spike_type'] = trimmed_spike_df['spike_type']
    merged_df['spike_magnitude'] = trimmed_spike_df['spike_magnitude']

    # Merge the weather data and the trimmed spike data
    # merged_df = pd.merge(weather_df, trimmed_spike_df)

    return merged_df



# def find_closest_rows_matrix(df, id_statistics, weather_list, date_list):
#     # Convert columns to NumPy arrays
#     id_stats_array = np.array(df['ID-statistics'].apply(eval).tolist())
#     weather_array = np.array(df['weather'].apply(eval).tolist())
#     date_sin_cos_array = df[['date_sin', 'date_cos']].values.astype(np.float64)

#     # Stack weather and date_sin_cos for parallel processing
#     weather_stack = np.array(weather_list)
#     date_stack = np.array(date_list)

#     # Calculate the Euclidean distance for each component in parallel
#     id_stats_distance = np.linalg.norm(id_stats_array - np.array(id_statistics), axis=1)
#     weather_distance_matrix = np.linalg.norm(weather_array[:, None, :] - weather_stack, axis=2)
#     date_distance_matrix = np.linalg.norm(date_sin_cos_array[:, None, :] - date_stack, axis=2)

#     # Total distance
#     total_distance_matrix = id_stats_distance[:, None] + weather_distance_matrix + date_distance_matrix

#     # Find the indices of the minimum distance for each day
#     closest_indices = np.argmin(total_distance_matrix, axis=0)

#     # Extract the closest rows' spike-related information
#     spike_nums = df.iloc[closest_indices]['spike_num'].values
#     spike_durations = df.iloc[closest_indices]['spike_durations'].apply(eval).tolist()
#     spike_types = df.iloc[closest_indices]['spike_type'].apply(eval).tolist()
#     spike_magnitudes = df.iloc[closest_indices]['spike_magnitudes'].apply(eval).tolist()
#     spike_times_intervals = df.iloc[closest_indices]['spike_times_intervals'].apply(eval).tolist()
    
#     return spike_nums, spike_durations, spike_types, spike_magnitudes, spike_times_intervals

# def get_spike_df_for_multiple_days(id_statistics, weather_list, date_list):
#     df = pd.read_csv('../../Data_processed/spike_tensors_x.csv')
    
#     # Get the closest rows for all days in one call
#     spike_nums, spike_durations, spike_types, spike_magnitudes, spike_times_intervals = find_closest_rows_matrix(df, id_statistics, weather_list, date_list)
    
#     all_days_spike_df = pd.DataFrame()
    
#     for day_idx in range(len(weather_list)):
#         spike_df = pd.DataFrame()
#         spike_df['time'] = pd.date_range('00:00:00', '23:30:00', freq='30T').time
#         spike_df['spike_type'] = 0
#         spike_df['spike_magnitude'] = 1.0

#         for i in range(spike_nums[day_idx]):
#             spike_start = spike_times_intervals[day_idx][i]
#             spike_duration = spike_durations[day_idx][i]
#             spike_end = spike_start + spike_duration

#             if spike_start + 1 < spike_end - 1:
#                 spike_center = np.random.randint(spike_start + 1, spike_end - 1)
#             else:
#                 spike_center = spike_start + (spike_duration // 2)

#             spike_df.loc[spike_start:spike_center-1, 'spike_type'] = 1
#             spike_df.loc[spike_center, 'spike_type'] = spike_types[day_idx][i]
#             spike_df.loc[spike_center, 'spike_magnitude'] = spike_magnitudes[day_idx][i]
#             spike_df.loc[spike_center+1:spike_end-1, 'spike_type'] = 4

#         all_days_spike_df = pd.concat([all_days_spike_df, spike_df], axis=0, ignore_index=True)

#     return all_days_spike_df

# def merge_weather_spike(weather_df, spike_df):
#     '''
#     Merge the weather and spike DataFrames

#     weather_df: DataFrame with the weather information
#     spike_df: DataFrame with the spike information

#     If the length of weather_df is less than spike_df, the spike_df is trimmed to match the length of weather_df
#     If the length of weather_df is greater than spike_df, the spike_df is replicated to match the length of weather_df

#     Returns a DataFrame with the weather and spike information merged
    
#     '''

#     if len(weather_df) < len(spike_df):
#         trimmed_spike_df = spike_df[:len(weather_df)]
#         merged_df = pd.concat([weather_df, trimmed_spike_df.reset_index(drop=True)], axis=1)
#     else:
#         replicate_times = len(weather_df) // len(spike_df) + (1 if len(weather_df) % len(spike_df) != 0 else 0)
#         # Replicate spike_df
#         replicated_spike_df = pd.concat([spike_df] * replicate_times, ignore_index=True)
#         # Trim the replicated_spike_df to match the length of weather_df
#         trimmed_replicated_spike_df = replicated_spike_df[:len(weather_df)]
#         # Merge the DataFrames
#         merged_df = pd.concat([weather_df.reset_index(drop=True), trimmed_replicated_spike_df.reset_index(drop=True)], axis=1)

#     merged_df.drop(columns=['time'], inplace=True)
#     return merged_df

# def date_to_sin_cos(date):
#     """Convert a date to its sine and cosine representation."""
#     day_of_year = date.timetuple().tm_yday
#     total_days = 365 + (1 if date.year % 4 == 0 else 0)  # Account for leap years
#     angle = 2 * np.pi * (day_of_year / total_days)
#     date_sin = np.sin(angle)
#     date_cos = np.cos(angle)
#     return date_sin, date_cos

# def generate_date_sin_cos_list(start_date, end_date):
#     """Generate a list of sine-cosine values for all dates between start_date and end_date."""
#     # Generate date range
#     date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
#     # Convert each date to its sine-cosine form
#     sin_cos_list = [date_to_sin_cos(date) for date in date_range]
    
#     return sin_cos_list

# def daily_weather_means(weather_df):
#     # Convert the 'tstp' column to datetime if it's not already
#     weather_df['tstp'] = pd.to_datetime(weather_df['tstp'])
    
#     # Extract the date from the timestamp
#     weather_df['date'] = weather_df['tstp'].dt.date
    
#     # Group by date and calculate the mean for each group
#     daily_means = weather_df.groupby('date').agg({
#         'temperature': 'mean',
#         'windSpeed': 'mean',
#         'humidity': 'mean'
#     }).reset_index()
    
#     # Convert the resulting DataFrame into a list of lists
#     weather_list = daily_means[['temperature', 'windSpeed', 'humidity']].values.tolist()
    
#     return weather_list

# def trim_and_merge_spike_weather(weather_df, spike_df):
#     # Convert 'tstp' and 'time' to datetime if they are not already
#     weather_df['tstp'] = pd.to_datetime(weather_df['tstp'])
    
#     # Extract the date part from 'tstp' for merging
#     weather_df['time'] = weather_df['tstp'].dt.time

#     # Get the start and end time of the weather data
#     weather_start_time = weather_df['time'].min()
#     weather_end_time = weather_df['time'].max()

#     # Convert 'time' in spike_df to datetime.time for comparison
#     spike_df['time'] = pd.to_datetime(spike_df['time'], format='%H:%M:%S').dt.time

#     # Trim the spike data to match the time range of the weather data
#     trimmed_spike_df = spike_df[(spike_df['time'] >= weather_start_time) & (spike_df['time'] <= weather_end_time)]

#     # Merge the weather data and the trimmed spike data
#     merged_df = pd.merge(weather_df, trimmed_spike_df, on='time', how='left')

#     return merged_df

# def generate_synthetic_energy(start_date_time = None, end_date_time = None, statistics = None, spike_hours = None, pre_spike_length = None, post_spike_length = None, model_medoid = None, model_m2s = None, device = None):
#     '''
#     Generate synthetic energy data

#     start_date_time: string with the start date and time in the format 'YYYY-MM-DD HH:MM:SS'
#     end_date_time: string with the end date and time in the format 'YYYY-MM-DD HH:MM:SS'

#     statistics: list with the statistics in the following order: mean, median, std, min, max, gradient

#     spike_hours: list of strings in the format 'HH:MM:SS'
#     spike_durations: list of integers
#     Should be the same length

#     Returns a DataFrame with the time, and the synthetic energy data

#     time_df
#     tstp       
#     2013-06-25 00:00:00   
#     2013-06-25 00:30:00
#     2013-06-25 01:00:00
#     ...

#     synthetic_energy
#     [0.1, 0.2, 0.15, ...]

#     '''
#     weather_df = utils.weather_info(start_date_time, end_date_time)
#     time_df = weather_df[['tstp']]
#     USER_DEFINED_SPIKE = False

#     if spike_hours and pre_spike_length and post_spike_length:
#         spike_magnitude = statistics[4]
#         spike_df = utils.get_spike_df_input(spike_hours, pre_spike_length, post_spike_length, spike_magnitude)
#         USER_DEFINED_SPIKE = True
#     # else:
#     #     # print('No spike information provided. Use random spikes.')
#     #     # spike_df = get_48_spike_data()
#     #     mean = statistics[0]
#     #     std  = statistics[2]
#     #     max = statistics[4]
#     #     statistics_df = pd.DataFrame({'mean': statistics[0], 'median': statistics[1], 'std': statistics[2], 'min': statistics[3], 'max': statistics[4], 'gradient': statistics[5]}, index=[0])
#     #     X = statistics_df.values
#     #     kmoid_cluster = joblib.load('../Data_preprocess/kmedoids_model.joblib')
#     #     cluster = kmoid_cluster.predict(X)[0]
#     #     month = int(start_date_time.split('-')[1])
#     #     spike_df = utils.get_spike_df(cluster, month, mean, std, max)
#     #     # Repeat the spike_df once
#     #     spike_df = pd.concat([spike_df, spike_df], ignore_index=True)
#     #     USER_DEFINED_SPIKE = False

#     if USER_DEFINED_SPIKE:
#         w_spike_df = utils.merge_weather_spike(weather_df, spike_df)
#     else:
#         date_list = generate_date_sin_cos_list(start_date_time, end_date_time)
#         weather_list = daily_weather_means(weather_df)
#         spike_df = get_spike_df_for_multiple_days(statistics, weather_list, date_list)
#         w_spike_df = trim_and_merge_spike_weather(weather_df, spike_df)
    
#     w_spike_df = utils.merge_weather_spike(weather_df, spike_df)
#     w_spike_df = utils.cyclic_time(w_spike_df)

#     w_spike_s_df = utils.merge_statistics(w_spike_df, statistics)

#     noise = torch.randn(1, len(w_spike_s_df), utils.get_m_latent_dim()).to(device)

#     weather_columns = ['temperature', 'windSpeed', 'humidity']
#     cluster_columns = ['kmedoid_clusters']
#     time_columns = ['date_sin', 'date_cos', 'time_sin', 'time_cos']
#     statistical_columns = ['mean', 'median', 'std',  'min', 'max', 'gradient']
#     spike_columns = ['spike_type', 'spike_magnitude']

#     weather = torch.tensor(w_spike_s_df[weather_columns].values, dtype=torch.float32).unsqueeze(0).to(device)
#     cluster = torch.tensor(w_spike_s_df[cluster_columns].values, dtype=torch.float32).unsqueeze(0).to(device).int()
#     time = torch.tensor(w_spike_s_df[time_columns].values, dtype=torch.float32).unsqueeze(0).to(device)
#     statistical = torch.tensor(w_spike_s_df[statistical_columns].values, dtype=torch.float32).unsqueeze(0).to(device)
#     spike = torch.tensor(w_spike_s_df[spike_columns].values, dtype=torch.float32).unsqueeze(0).to(device).int()

#     spike_type = spike[:, :, 0:1].int().to(device)
#     spike_magnitude = spike[:, :, 1:].to(device)

#     h = model_medoid.lstm_decoder(noise, weather, cluster, time)
#     # Add noise to the hidden state
#     h = h + torch.randn(h.shape).to(device) * 0.1

#     latent_space = model_m2s.lstm_encoder(h)
#     mu, logvar = latent_space.chunk(2, dim=2)
#     z = model_m2s.reparameterize(mu, logvar)
#     h = model_m2s.lstm_decoder(z, weather, cluster, time, statistical, spike_type, spike_magnitude)

#     synthetic_energy = h.squeeze().detach().cpu().numpy()
#     min_energy = statistical.squeeze().detach().cpu().numpy()[0][3]
#     gradient = statistical.squeeze().detach().cpu().numpy()[0][-1]
#     enhanced_synthetic_energy = utils.enhanced_energy_profile(spike_type, synthetic_energy, gradient, min_energy)

#     return time_df, synthetic_energy, enhanced_synthetic_energy

def generate_spike_type(cluster, month):
    spike_type_prob_df = pd.read_csv('../../Data_processed/Spike_type_cluster_month.csv')
    temp_df = spike_type_prob_df[spike_type_prob_df['kmedoid_clusters'] == cluster]
    selected_spike_type = np.random.choice(temp_df['spike_type'], p=temp_df[f'month_{month}'])
    return selected_spike_type

def generate_spike_time(cluster, month, type):
    spike_time_prob_df = pd.read_csv('../../Data_processed/Spike_time_type_cluster_month.csv')
    temp_df = spike_time_prob_df[(spike_time_prob_df['kmedoid_clusters'] == cluster) & (spike_time_prob_df['spike_type'] == type)]
    selected_spike_time = np.random.choice(temp_df['time'], p=temp_df[f'month_{month}'])
    return selected_spike_time

def generate_synthetic_energy_t(start_date_time = None, end_date_time = None, statistics = None, spike_hours = None, pre_spike_length = None, post_spike_length = None, model_medoid = None, model_m2s = None, device = None):
    '''
    Generate synthetic energy data

    start_date_time: string with the start date and time in the format 'YYYY-MM-DD HH:MM:SS'
    end_date_time: string with the end date and time in the format 'YYYY-MM-DD HH:MM:SS'

    statistics: list with the statistics in the following order: mean, median, std, min, max, gradient

    spike_hours: list of strings in the format 'HH:MM:SS'
    spike_durations: list of integers
    Should be the same length

    Returns a DataFrame with the time, and the synthetic energy data
    '''
    weather_df = weather_info(start_date_time, end_date_time)
    time_df = weather_df[['tstp']]

    if spike_hours and pre_spike_length and post_spike_length:
        spike_magnitude = statistics[4]
        spike_df = get_spike_df_input(spike_hours, pre_spike_length, post_spike_length, spike_magnitude)
        w_spike_df = merge_weather_spike(weather_df, spike_df)
    else:
        mean = statistics[0]
        std  = statistics[2]
        max = statistics[4]
        statistics_df = pd.DataFrame({'mean': statistics[0], 'median': statistics[1], 'std': statistics[2], 'min': statistics[3], 'max': statistics[4], 'gradient': statistics[5]}, index=[0])
        X = statistics_df.values
        kmoid_cluster = joblib.load('../Data_preprocess/kmedoids_model.joblib')
        cluster = kmoid_cluster.predict(X)[0]
        month = int(start_date_time.split('-')[1])

        start_date_str = start_date_time.split()[0]
        end_date_str = end_date_time.split()[0]

        # Convert the date strings to datetime objects
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        # Calculate the difference in days
        num_days = (end_date - start_date).days + 1  # Including both start and end dates

        spike_df_days = []
        
        for i in range(num_days):
            spike_df = get_spike_df(cluster, month, mean, std, max)
            spike_df_days.append(spike_df)

        spike_df_days = pd.concat(spike_df_days, ignore_index=True)
        spike_df_days.to_csv('spike_df_days.csv', index=False)
        
        spike_df_days = add_missing_pre_post_spikes(spike_df_days)
        spike_df_days.to_csv('spike_df_days_added.csv', index=False)


        w_spike_df = trim_and_merge_spike_weather(weather_df, spike_df_days)
        w_spike_df.to_csv('w_spike_df.csv', index=False)
        
    w_spike_df = cyclic_time(w_spike_df)

    w_spike_s_df = merge_statistics(w_spike_df, statistics)

    noise = torch.randn(1, len(w_spike_s_df), get_m_latent_dim()).to(device)

    weather_columns = ['temperature', 'windSpeed', 'humidity']
    cluster_columns = ['kmedoid_clusters']
    time_columns = ['date_sin', 'date_cos', 'time_sin', 'time_cos']
    statistical_columns = ['mean', 'median', 'std',  'min', 'max', 'gradient']
    spike_columns = ['spike_type', 'spike_magnitude']

    weather = torch.tensor(w_spike_s_df[weather_columns].values, dtype=torch.float32).unsqueeze(0).to(device)
    cluster = torch.tensor(w_spike_s_df[cluster_columns].values, dtype=torch.float32).unsqueeze(0).to(device).int()
    time = torch.tensor(w_spike_s_df[time_columns].values, dtype=torch.float32).unsqueeze(0).to(device)
    statistical = torch.tensor(w_spike_s_df[statistical_columns].values, dtype=torch.float32).unsqueeze(0).to(device)
    spike = torch.tensor(w_spike_s_df[spike_columns].values, dtype=torch.float32).unsqueeze(0).to(device).int()

    spike_type = spike[:, :, 0:1].int().to(device)
    spike_magnitude = spike[:, :, 1:].to(device)

    h = model_medoid.lstm_decoder(noise, weather, cluster, time)
    # Add noise to the hidden state
    h = h + torch.randn(h.shape).to(device) * 0.1

    latent_space = model_m2s.lstm_encoder(h)
    mu, logvar = latent_space.chunk(2, dim=2)
    z = model_m2s.reparameterize(mu, logvar)
    h = model_m2s.lstm_decoder(z, weather, cluster, time, statistical, spike_type, spike_magnitude)

    synthetic_energy = h.squeeze().detach().cpu().numpy()

    min_energy = statistical.squeeze().detach().cpu().numpy()[0][3]
    gradient = statistical.squeeze().detach().cpu().numpy()[0][-1]
    
    enhanced_synthetic_energy = enhanced_energy_profile(spike_type, synthetic_energy, gradient, min_energy)

    return time_df, synthetic_energy, enhanced_synthetic_energy