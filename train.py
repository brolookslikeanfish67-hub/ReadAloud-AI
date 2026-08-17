# =====================================================================
# Orchestration Runtime with Cross-Attention Speaker Conditioning
# =====================================================================

import os
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

# Import your custom modular files from the workspace directory
from dataset import BestEmiliaStreamingPipeline, emilia_smart_collate_fn
from tokenizer import ContinuousAudioTokenizer
from model import HighFidelityOmniVoiceEngine

def save_system_checkpoint(tokenizer, model, optimizer, step_count, filepath="omnivoice_cloner_checkpoint.pt"):
    """Saves complete model weights, cloner parameters, and tracking metrics to disk."""
    checkpoint_state = {
        'step_count': step_count,
        'tokenizer_state_dict': tokenizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint_state, filepath)
    print(f"[+] Checkpoint Auto-Saved Successfully: '{filepath}' at step {step_count}")

def load_system_checkpoint(tokenizer, model, optimizer, filepath="omnivoice_cloner_checkpoint.pt"):
    """Resumes structural weights and tracking state metrics from a saved checkpoint."""
    if os.path.exists(filepath):
        print(f"[*] Found existing checkpoint asset file: '{filepath}'. Resuming training...")
        checkpoint_state = torch.load(filepath, map_location=next(model.parameters()).device)
        tokenizer.load_state_dict(checkpoint_state['tokenizer_state_dict'])
        model.load_state_dict(checkpoint_state['model_state_dict'])
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        return checkpoint_state['step_count']
    print("[*] No prior checkpoint files detected on disk. Initializing a clean training run.")
    return 0

def run_zero_shot_training_framework(max_steps=1000, checkpoint_interval=200, batch_size=2):
    """
    Master execution runtime wrapper optimized for real-world continuous data streaming,
    cross-attention reference style conditioning, and zero-shot voice synthesis.
    """
    # 1. Device Hardware Routing Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Activating processing grid on target hardware tool: {device}")

    # 2. Match Audio Framework Constraints
    target_sample_rate = 16000
    max_audio_samples = 48000 # 3.0 seconds at 16kHz
    
    # 3. Instantiate the Multi-Threaded Live Streaming Pipeline
    print("[*] Spawning network prefetch pipelines for target text strings...")
    best_dataset = BestEmiliaStreamingPipeline(
        hf_token=None, 
        sample_rate=target_sample_rate,
        max_audio_samples=max_audio_samples,
        prefetch_buffer_size=32 
    )
    
    data_loader = DataLoader(
        best_dataset, 
        batch_size=batch_size, 
        collate_fn=emilia_smart_collate_fn
    )

    # 4. Core Network Component Assemblies
    print("[*] Building engine matrix networks...")
    audio_tokenizer = ContinuousAudioTokenizer(embed_dim=256).to(device)
    omni_voice_core = HighFidelityOmniVoiceEngine(embed_dim=256, max_diffusion_steps=50).to(device)

    # Combine architectural components inside a unified AdamW optimizer tracking sweep
    combined_params = list(audio_tokenizer.parameters()) + list(omni_voice_core.parameters())
    optimizer = torch.optim.AdamW(combined_params, lr=1e-4, weight_decay=1e-2)

    # 5. Checkpoint Lifecycle Verification
    global_step = load_system_checkpoint(audio_tokenizer, omni_voice_core, optimizer)

    # 6. MASTER ZERO-SHOT CLONING OPTIMIZATION RUN
    print(f"\n[+] Entering Voice Cloning Training Loops (Targeting {max_steps} Global Steps):")
    print("=" * 95)
    
    audio_tokenizer.train()
    omni_voice_core.train()

    stream_iterator = iter(data_loader)

    while global_step < max_steps:
        try:
            waveforms, text_tokens = next(stream_iterator)
        except StopIteration:
            print("[*] Live stream cycle completed. Refreshing stream connections...")
            stream_iterator = iter(data_loader)
            waveforms, text_tokens = next(stream_iterator)
        except Exception:
            continue

        waveforms = waveforms.to(device)
        text_tokens = text_tokens.to(device)

        # To train the zero-shot encoder, we extract a slice of the waveform 
        # to act as our speaker reference acoustic clip.
        # Here, the current audio serves as both target and speaker identity reference.
        reference_waveforms = waveforms.clone()

        optimizer.zero_grad()

        # --- SUB-LOSS A: SYMMETRIC VAE AUDIO AUTOENCODER ---
        reconstructed_audio, mu, logvar = audio_tokenizer(waveforms)
        recon_loss = F.mse_loss(reconstructed_audio, waveforms)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / waveforms.size(0)

        # --- SUB-LOSS B: CROSS-ATTENTION DIFFUSION CLONING SPEECH ENGINE ---
        with torch.no_grad():
            latent_z, _, _ = audio_tokenizer.encoder(waveforms)
            
        timesteps = torch.randint(0, omni_voice_core.max_steps, (waveforms.size(0),), device=device)
        noise_targets = torch.randn_like(latent_z)
        noisy_latents = latent_z + noise_targets * 0.1 

        # Predict noise parameters based on BOTH text tokens and speaker reference templates
        predicted_noise = omni_voice_core(text_tokens, reference_waveforms, noisy_latents, timesteps)
        diffusion_loss = F.mse_loss(predicted_noise, noise_targets)

        # --- JOINT MULTI-OBJECTIVE WEIGHTED BACKPROPAGATION PASS ---
        total_loss = diffusion_loss + (0.1 * recon_loss) + (0.005 * kl_loss)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(combined_params, max_norm=1.0)
        optimizer.step()

        global_step += 1

        if global_step % 10 == 0 or global_step == 1:
            print(f" Step [{global_step:05d}/{max_steps:05d}] | Total Loss: {total_loss.item():.5f} | Diffusion: {diffusion_loss.item():.5f} | VAE Recon: {recon_loss.item():.5f}")

        if global_step % checkpoint_interval == 0:
            save_system_checkpoint(audio_tokenizer, omni_voice_core, optimizer, global_step)

    # 7. PRODUCTION ZERO-SHOT CLONING SPEECH INFERENCE TARGET RUN
    print("\n" + "=" * 95)
    print("[+] Model Optimization Complete. Executing Active Zero-Shot Voice Cloning Inference...")
    print("=" * 95)
    
    cloning_prompt = "Zero shot cross attention speech models capture target pitch dynamics fluidly."
    print(f"[*] Target Generation Phrase: '{cloning_prompt}'")
    
    encoded_bytes = [b for b in cloning_prompt.encode('utf-8')]
    inference_tokens = torch.tensor([encoded_bytes], dtype=torch.long, device=device)
    
    # SETUP TARGET VOICE TEMPLATE FOR THE ZERO-SHOT CLONE
    # In live usage, swap this mock tensor out for a real 3-second .wav file of any human speaker.
    # example: mock_reference_voice, _ = torchaudio.load("target_speaker_voice.wav")
    print("[*] Registering target voice profile template...")
    mock_reference_voice = torch.randn(1, 1, max_audio_samples).to(device) * 0.1
    
    audio_tokenizer.eval()
    omni_voice_core.eval()
    
    target_compressed_frames = max_audio_samples // 64 

    with torch.no_grad():
        print("[*] Running reverse diffusion denoising trajectory maps across speaker style fields...")
        generated_latents = omni_voice_core.generate_latent_trajectory(
            inference_tokens, 
            mock_reference_voice,
            num_target_frames=target_compressed_frames
        )
        
        print("[*] Projecting pristine cloned code states through the structural decoder channels...")
        synthesized_waveform = audio_tokenizer.decoder(generated_latents)

    final_output_tensor = synthesized_waveform.squeeze(0).cpu()
    output_wav_name = "cloned_zero_shot_output.wav"
    
    print(f"[*] Exporting physical cloned soundwave asset file: '{output_wav_name}'...")
    torchaudio.save(output_wav_name, final_output_tensor, sample_rate=target_sample_rate)
    print("[+] Complete. Voice cloning execution loop completed successfully.")


if __name__ == "__main__":
    run_zero_shot_training_framework(max_steps=1000, checkpoint_interval=200, batch_size=2)
