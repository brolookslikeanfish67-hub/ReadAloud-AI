# =====================================================================
# Master Orchestration Runtime for Training and Speech Synthesis
# =====================================================================

import os
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

# Import your previously written custom architectural files
from dataset import InsaneEmiliaDataset, emilia_smart_collate_fn
from tokenizer import ContinuousAudioTokenizer
from model import HighFidelityOmniVoiceEngine

def execute_complete_pipeline(dataset_directory="./emilia_raw_data", epochs=3, batch_size=2):
    """
    Main orchestration function handling initialization, dataset compilation,
    joint network optimization passes, and final zero-shot speech synthesis.
    """
    # 1. Device Hardware Discovery Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing processing architecture execution grid on tool: {device}")

    # 2. Data Infrastructure Pipeline Initialization
    # Enforces uniform frame sizes: 16,000Hz * 3 seconds = 48,000 max samples
    target_sample_rate = 16000
    max_audio_samples = 48000
    
    print("[*] Building multi-lingual dataset collection frameworks...")
    emilia_dataset = InsaneEmiliaDataset(
        dataset_root=dataset_directory, 
        sample_rate=target_sample_rate, 
        max_audio_samples=max_audio_samples
    )
    
    data_loader = DataLoader(
        emilia_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=emilia_smart_collate_fn
    )

    # 3. Model Subsystem Instantiations
    print("[*] Instantiating neural components and structural modules...")
    audio_tokenizer = ContinuousAudioTokenizer(embed_dim=256).to(device)
    omni_voice_core = HighFidelityOmniVoiceEngine(embed_dim=256, max_diffusion_steps=50).to(device)

    # Combine all parameters into a unified AdamW optimizer tracking routine
    combined_parameters = list(audio_tokenizer.parameters()) + list(omni_voice_core.parameters())
    optimizer = torch.optim.AdamW(combined_parameters, lr=1e-4, weight_decay=1e-2)

    # 4. COMPOSITE TRAINING OPTIMIZATION LOOP
    print(f"\n[+] Launching System Optimization Sequence ({epochs} Total Epochs):")
    print("=" * 85)
    
    for epoch in range(1, epochs + 1):
        audio_tokenizer.train()
        omni_voice_core.train()
        
        epoch_diffusion_loss = 0.0
        epoch_recon_loss = 0.0
        
        for batch_idx, (waveforms, text_tokens) in enumerate(data_loader):
            waveforms = waveforms.to(device)     # Shape: [Batch, 1, Audio_Samples]
            text_tokens = text_tokens.to(device) # Shape: [Batch, Sequence_Len]
            
            optimizer.zero_grad()

            # --- SUB-OBJECTIVE A: CONTINUOUS TOKENIZER VAE TRAIN ---
            # Pass physical audio through the VAE encoder/decoder blocks
            reconstructed_audio, mu, logvar = audio_tokenizer(waveforms)
            
            # Reconstruction Loss: Ensures output sounds like input
            recon_loss = F.mse_loss(reconstructed_audio, waveforms)
            # KL Divergence: Smooths out hidden token clusters
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / waveforms.size(0)

            # --- SUB-OBJECTIVE B: TEXT-CONDITIONED LATENT DIFFUSION TRAIN ---
            # Re-extract clean latent variables via the encoder distribution pass
            with torch.no_grad():
                latent_z, _, _ = audio_tokenizer.encoder(waveforms)
                
            # Sample random execution index states across diffusion timelines
            timesteps = torch.randint(0, omni_voice_core.max_steps, (waveforms.size(0),), device=device)
            
            # Generate random target shapes matching the latent layout dimensions
            noise_targets = torch.randn_like(latent_z)
            
            # Inject forward-diffusion noise onto clean audio latent tensors
            # Operates directly under the 'denoise = False' training parameters
            noisy_latents = latent_z + noise_targets * 0.1
            
            # Predict noise artifacts across text prompt boundaries
            predicted_noise = omni_voice_core(text_tokens, noisy_latents, timesteps)
            diffusion_loss = F.mse_loss(predicted_noise, noise_targets)

            # --- COMPOSITE TOTAL BACKPROPAGATION PASS ---
            # Weighted loss compilation combining acoustic features and language features
            total_loss = diffusion_loss + (0.1 * recon_loss) + (0.005 * kl_loss)
            
            total_loss.backward()
            optimizer.step()

            # Accumulate logs for console tracking
            epoch_diffusion_loss += diffusion_loss.item()
            epoch_recon_loss += recon_loss.item()

        avg_diff = epoch_diffusion_loss / len(data_loader)
        avg_rec = epoch_recon_loss / len(data_loader)
        print(f" Epoch {epoch:02d}/{epochs:02d} | Avg Diffusion Loss: {avg_diff:.5f} | Avg VAE Reconstruction: {avg_rec:.5f}")

    # 5. ZERO-SHOT SPEECH SYNTHESIS GENERATION INFERENCE
    print("\n" + "=" * 85)
    print("[+] Model Optimization Complete. Initializing Zero-Shot Generation Loop...")
    print("=" * 85)
    
    # Define an unconditioned multi-lingual multi-character sentence prompt block
    target_prompt = "Hello world! Building a production grade text to speech rival system step by step. 这是一个全功能系统。"
    print(f"[*] Target Universal Prompt: '{target_prompt}'")
    
    # Process string into character token format using dataset method logic
    encoded_bytes = [b for b in target_prompt.encode('utf-8')]
    inference_tokens = torch.tensor([encoded_bytes], dtype=torch.long, device=device)
    
    audio_tokenizer.eval()
    omni_voice_core.eval()
    
    # Estimate compressed frame dimensions (~7.5 Hz framework mapping ratio)
    target_compressed_frames = max_audio_samples // 64 # Matches tokenizer compression factor
    
    with torch.no_grad():
        print("[*] Running reverse diffusion denoising chain across text spaces...")
        # Step A: Synthesize structural latents out of pure white noise parameters
        denoised_latents = omni_voice_core.generate_latent_trajectory(
            inference_tokens, 
            num_target_frames=target_compressed_frames
        )
        
        print("[*] Passing denoised code trajectories through the audio decoder array...")
        # Step B: Project code configurations through the VAE decoder to create an analog wave
        generated_waveform = audio_tokenizer.decoder(denoised_latents)
        
    # Move tensor data safely back to the CPU memory banks for file writing
    output_waveform = generated_waveform.squeeze(0).cpu()
    output_filename = "omnivoice_rival_output.wav"
    
    print(f"[*] Exporting final physical audio asset file to disk: '{output_filename}'...")
    torchaudio.save(output_filename, output_waveform, sample_rate=target_sample_rate)
    print("[+] Process Complete! File generated successfully with proper dimensions.")


if __name__ == "__main__":
    # Execute the master system routine pipeline
    execute_complete_pipeline(dataset_directory="./emilia_raw_data", epochs=3, batch_size=2)
