import logging
import torch
import network
import json
import torch.optim as optim
import torch.nn as nn
import utils
import z_networks


def WGAN_hp_search(gen_parameter_range, dis_parameter_range, train_loader, test_loader, Generator, Discriminator, device):
    best_gen_loss = float('inf')
    best_dis_loss = float('-inf')
    best_gen_config = {}
    best_dis_config = {}

    for g_key in gen_parameter_range:
        best_gen_config[g_key] = gen_parameter_range[g_key][0]
    for d_key in dis_parameter_range:
        best_dis_config[d_key] = dis_parameter_range[d_key][0]

    for g_key in gen_parameter_range:
        logging.info(f'Optimizing Generator parameter {g_key}')
        for g_value in gen_parameter_range[g_key]:
            current_gen_config = best_gen_config.copy()
            current_gen_config[g_key] = g_value
            logging.info(f'Current Generator configuration: {current_gen_config}')

            gen = Generator(**current_gen_config).to(device)

            for d_key in dis_parameter_range:
                logging.info(f'Optimizing Discriminator parameter {d_key} with Generator parameter {g_key} = {g_value}')
                for d_value in dis_parameter_range[d_key]:
                    current_dis_config = best_dis_config.copy()
                    current_dis_config[d_key] = d_value
                    logging.info(f'Optimizing Discriminator Parameter {d_key} with value {d_value}')

                    dis = Discriminator(**current_dis_config).to(device)

                    gen_optimizer = torch.optim.RMSprop(gen.parameters(), lr=1e-4)
                    dis_optimizer = torch.optim.RMSprop(dis.parameters(), lr=1e-4)

                    patience = 0

                    best_epoch_g_loss = float('inf')
                    best_epoch_d_loss = float('-inf')

                    for epoch in range(20):
                        gen_total_loss = 0
                        dis_total_loss = 0

                        for i, (weather, cluster, time, statistical, spike, energy) in enumerate(train_loader):
                            weather = weather.to(device)
                            cluster = cluster.to(device).int()
                            time = time.to(device)
                            statistical = statistical.to(device)
                            spike_type = spike[:, :, 0:1].to(dtype=torch.int32).to(device)
                            spike_magnitude = spike[:, :, 1:].to(dtype=torch.float32).to(device)
                            energy = energy.to(device)

                            gen.train()
                            dis.train()
                            # Train Discriminator
                            dis_optimizer.zero_grad()
                            real = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, energy)
                            fake = gen(weather, cluster, time, statistical, spike_type, spike_magnitude)
                            dis_fake = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, fake)
                            dis_loss = torch.mean(dis_fake) - torch.mean(real)
                            dis_loss.backward()
                            dis_optimizer.step()

                            # Weight Clipping
                            for p in dis.parameters():
                                p.data.clamp_(-0.1, 0.1)

                            if i % 5 == 0:  # Train generator every 5 iterations
                                # Train Generator
                                gen_optimizer.zero_grad()
                                fake = gen(weather, cluster, time, statistical, spike_type, spike_magnitude)
                                dis_fake = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, fake)
                                gen_loss = -torch.mean(dis_fake)
                                gen_loss.backward()
                                gen_optimizer.step()
                        
                        for i, (weather, cluster, time, statistical, spike, energy) in enumerate(test_loader):
                            weather = weather.to(device)
                            cluster = cluster.to(device).int()
                            time = time.to(device)
                            statistical = statistical.to(device)
                            spike_type = spike[:, :, 0:1].to(dtype=torch.int32).to(device)
                            spike_magnitude = spike[:, :, 1:].to(dtype=torch.float32).to(device)
                            energy = energy.to(device)
                            
                            gen.eval()
                            dis.eval()
                            fake = gen(weather, cluster, time, statistical, spike_type, spike_magnitude)
                            dis_fake = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, fake)
                            dis_real = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, energy)
                            gen_loss_t = -torch.mean(dis_fake)
                            dis_loss_t = torch.mean(dis_fake) - torch.mean(dis_real)
                            gen_total_loss += gen_loss_t.item()
                            dis_total_loss += dis_loss_t.item()

                        epoch_gen_loss = gen_total_loss / len(test_loader)
                        epoch_dis_loss = dis_total_loss / len(test_loader)
                        logging.info(f'Epoch {epoch}, Discriminator Loss: {epoch_dis_loss}, Generator Loss: {epoch_gen_loss}')
                    
                        if epoch_gen_loss < best_epoch_g_loss:
                            best_epoch_g_loss = epoch_gen_loss
                            patience = 0
                        elif epoch_dis_loss > best_epoch_d_loss:
                            best_epoch_d_loss = epoch_dis_loss
                            patience = 0
                        else:
                            patience += 1


                        if patience == 5:
                            break
                    
                    if best_epoch_g_loss < best_gen_loss and best_epoch_d_loss > best_dis_loss:
                        best_gen_loss = best_epoch_g_loss
                        best_dis_loss = best_epoch_d_loss
                        best_gen_config = current_gen_config
                        best_dis_config = current_dis_config
                        logging.info(f'New best Generator configuration: {best_gen_config}')
                        logging.info(f'New best Discriminator configuration: {best_dis_config}')

    logging.info(f'Best Generator configuration: {best_gen_config}')
    logging.info(f'Best Discriminator configuration: {best_dis_config}')
    return best_gen_config, best_dis_config

def train_WGAN(gen, dis, train_loader, test_loader, device, num_epochs, clip_value=0.01):

    best_gen_loss = float('inf')
    best_dis_loss = float('-inf')

    gen_optimizer = torch.optim.RMSprop(gen.parameters(), lr=3e-5)
    dis_optimizer = torch.optim.RMSprop(dis.parameters(), lr=3e-5)

    for epoch in range(num_epochs):
        gen_total_loss = 0
        dis_total_loss = 0

        gen.train()
        dis.train()

        for i, (weather, cluster, time, statistical, spike, energy) in enumerate(train_loader):
            weather = weather.to(device)
            cluster = cluster.to(device).int()
            time = time.to(device)
            statistical = statistical.to(device)
            spike_type = spike[:, :, 0:1].to(dtype=torch.int32).to(device)
            spike_magnitude = spike[:, :, 1:].to(dtype=torch.float32).to(device)
            energy = energy.to(device)

            # Train Discriminator
            dis_optimizer.zero_grad()
            real = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, energy)
            fake = gen(weather, cluster, time, statistical, spike_type, spike_magnitude)
            dis_fake = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, fake)
            dis_loss = torch.mean(dis_fake) - torch.mean(real)
            dis_loss.backward()
            dis_optimizer.step()

            # Weight Clipping
            for p in dis.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if i % 5 == 0:  # Train generator every 5 iterations
                # Train Generator
                gen_optimizer.zero_grad()
                fake = gen(weather, cluster, time, statistical, spike_type, spike_magnitude)
                dis_fake = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, fake)
                gen_loss = -torch.mean(dis_fake)
                gen_loss.backward()
                gen_optimizer.step()

            # Logging intermediate results
            if (i+1) % 1000 == 0:
                logging.info(f'Iteration {i}, Discriminator Loss: {dis_loss.item()}, Generator Loss: {gen_loss.item()}')

        gen.eval()
        dis.eval()

        # Validation loop
        with torch.no_grad():
            for i, (weather, cluster, time, statistical, spike, energy) in enumerate(test_loader):
                weather = weather.to(device)
                cluster = cluster.to(device).int()
                time = time.to(device)
                statistical = statistical.to(device)
                spike_type = spike[:, :, 0:1].to(dtype=torch.int32).to(device)
                spike_magnitude = spike[:, :, 1:].to(dtype=torch.float32).to(device)
                energy = energy.to(device)

                fake = gen(weather, cluster, time, statistical, spike_type, spike_magnitude)
                dis_fake = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, fake)
                dis_real = dis(weather, cluster, time, statistical, spike_type, spike_magnitude, energy)
                gen_loss_t = -torch.mean(dis_fake)
                dis_loss_t = torch.mean(dis_fake) - torch.mean(dis_real)
                gen_total_loss += gen_loss_t.item()
                dis_total_loss += dis_loss_t.item()

        epoch_gen_loss = gen_total_loss / len(test_loader)
        epoch_dis_loss = dis_total_loss / len(test_loader)
        logging.info(f'Epoch {epoch}, Discriminator Loss: {epoch_dis_loss}, Generator Loss: {epoch_gen_loss}')

        if (epoch_gen_loss < best_gen_loss or epoch_dis_loss > best_dis_loss) and epoch > 10:
            print('New Best Model')
            best_gen_loss = epoch_gen_loss
            best_dis_loss = epoch_dis_loss
            best_gen = gen
            best_disc = dis

    logging.info('Training complete.')
    return best_gen, best_disc

def train_s_vae(model, optimizer, dataloader, device, num_epochs):
    best_loss = float('inf')
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for idx, (weather, cluster, time, statistical, spike, real_energy) in enumerate(dataloader):
            weather = weather.to(device)
            cluster = cluster.to(device).int()
            time = time.to(device)
            statistical = statistical.to(device)
            spike_type = spike[:, :, 0:1].to(dtype=torch.int32).to(device)
            spike_magnitude = spike[:, :, 1:].to(dtype=torch.float32).to(device)
            real_energy = real_energy.to(device)
            noise = torch.randn(real_energy.shape[0], real_energy.shape[1], 1).to(device)

            optimizer.zero_grad()
            synthetic_profile, mu, logvar = model(noise, weather, cluster, time, statistical, spike_type, spike_magnitude)

            loss = utils.vae_loss_function(synthetic_profile, real_energy, mu, logvar)
            loss += 10 * utils.gradient_loss(synthetic_profile, statistical)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            if (idx + 1) % 10000 == 0:
                logging.info(f'Step {idx + 1} / {len(dataloader)}, Loss: {loss.item()}')

        epoch_loss = total_loss / num_batches
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1

        if patience == 3:
            logging.info('Early stopping triggered')
            break

    logging.info(f'Best loss achieved: {best_loss}')
    return model, best_loss

criterion_mse = nn.MSELoss()
criterion_spike_num = nn.CrossEntropyLoss()

def s_VAE_hsearch(model, space, dataloader, device, num_epochs):
    best_loss = float('inf')
    best_config = {key: space[key][0] for key in space} # Initialize the best_config with the first values from the space

    for key in space:
        logging.info(f"Optimizing hyperparameter: {key}")
        
        for value in space[key]:
            current_config = best_config.copy()
            current_config[key] = value

            logging.info(f"Testing configuration: {current_config}")

            model = z_networks.s_VAE(**current_config).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            current_loss = s_VAE_train_for_search(model, num_epochs, dataloader, optimizer, device)
        
            if current_loss < best_loss:
                best_loss = current_loss
                best_config[key] = value

        logging.info(f'Best value for {key}: {best_config[key]}')
        logging.info(f'Current best configuration: {best_config}')
    
    logging.info(f'Best configuration: {best_config}')
    logging.info(f'Best loss: {best_loss}')

    return best_config, best_loss

def s_VAE_train_for_search(model, num_epochs, dataloader, optimizer, device):
    best_loss = float('inf')
    model.train()
    patience = 0
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for _, (spike_nums, spike_durations, spike_magnitudes, ID_statistics, weather, date_sin, date_cos, time_sin_cos) in enumerate(dataloader):
            spike_nums = spike_nums.to(device)
            spike_durations = spike_durations.to(device)
            spike_magnitudes = spike_magnitudes.to(device)
            ID_statistics = ID_statistics.to(device)
            weather = weather.to(device)
            date_sin = date_sin.to(device)
            date_cos = date_cos.to(device)
            time_sin_cos = time_sin_cos.to(device)

            optimizer.zero_grad()
            
            spike_num_recon, spike_durations_recon, spike_magnitudes_recon, time_sin_cos_recon, mu, logvar = model(
                spike_nums, spike_durations, spike_magnitudes, ID_statistics, weather, date_sin, date_cos, time_sin_cos
            )

            # Compute spike_num_loss
            spike_num_label = spike_nums.view(-1)
            spike_num_loss = criterion_spike_num(spike_num_recon, spike_num_label)

            # Masking and spike_durations_loss
            mask_duration = spike_durations > 0
            spike_durations = spike_durations[mask_duration]
            spike_durations_recon = spike_durations_recon[mask_duration]
            spike_duration_loss = criterion_mse(spike_durations_recon, spike_durations)

            # Masking and spike_magnitudes_loss
            mask_magnitude = spike_magnitudes > 0
            spike_magnitudes = spike_magnitudes[mask_magnitude]
            spike_magnitudes_recon = spike_magnitudes_recon[mask_magnitude]
            spike_magnitude_loss = criterion_mse(spike_magnitudes_recon, spike_magnitudes)

            # Masking and time_loss
            mask_time = time_sin_cos.sum(dim=-1) != 0  # Masking based on non-zero sum in the last dimension
            time_sin_cos = time_sin_cos[mask_time]
            time_sin_cos_recon = time_sin_cos_recon[mask_time]
            time_loss = criterion_mse(time_sin_cos_recon, time_sin_cos)
            
            # Compute KL Divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # kl_div /= spike_nums.size(0)  # Normalize by batch size
            
            # Total loss
            loss = spike_num_loss + spike_duration_loss + spike_magnitude_loss + time_loss + kl_div

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        epoch_loss = total_loss / num_batches
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1

        if patience > 3:
            break

    return best_loss