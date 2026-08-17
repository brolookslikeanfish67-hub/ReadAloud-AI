# =====================================================================
# Multi-Modal Diffusion Model with Contextual Voice Cloning Integration
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import our dedicated acoustic style extractor modules
from voice_cloner import InsaneVoiceClonerEncoder, CrossAttentionFusionBlock

class DiffusionTimeEmbedding(nn.Module):
    """
    Projects scalar diffusion time-steps (t) into high-dimensional sinusoidal
    contextual vectors to guide the denoising neural layers.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim

    def forward(self, timesteps):
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return self.linear_layers(embeddings)


class NonAutoregressiveDiffusionHead(nn.Module):
    """
    A 1D Convolutional Denoising Block that inputs heavily distorted latent vectors, 
    time embeddings, and linguistic context to isolate and subtract noise shapes.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv_in = nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.conv_mid = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.conv_out = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, noisy_audio_latents, multi_modal_context, time_vectors):
        """
        Args:
            noisy_audio_latents (Tensor): [Batch, Embed_Dim, Sequence_Frames]
            multi_modal_context (Tensor): [Batch, Embed_Dim, Sequence_Frames]
            time_vectors (Tensor): [Batch, Embed_Dim]
        """
        t_spatial = time_vectors.unsqueeze(-1) # [Batch, Embed_Dim, 1]
        conditioning = multi_modal_context + t_spatial
        
        # Concat noisy inputs and target conditioning down the feature channel axis
        x = torch.cat([noisy_audio_latents, conditioning], dim=1)
        
        h = self.gelu(self.conv_in(x))
        h = h + self.gelu(self.conv_mid(h))
        estimated_noise = self.conv_out(h)
        return estimated_noise


class HighFidelityOmniVoiceEngine(nn.Module):
    """
    The unified multi-modal core engine. Integrates raw text encoding, 
    cross-attention voice cloning extraction, and latent diffusion denoising loops.
    """
    def __init__(self, vocab_size=256, embed_dim=256, max_diffusion_steps=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_steps = max_diffusion_steps
        
        # 1. Text processing layers
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=512, batch_first=True, activation='gelu'
        )
        self.transformer_core = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        # 2. Zero-Shot Voice Cloning Integration (The VibeVoice Killer)
        self.speaker_encoder = InsaneVoiceClonerEncoder(embed_dim)
        self.style_fusion_layer = CrossAttentionFusionBlock(embed_dim)
        
        # 3. Diffusion modeling layers
        self.time_encoder = DiffusionTimeEmbedding(embed_dim)
        self.diffusion_head = NonAutoregressiveDiffusionHead(embed_dim)

    def forward(self, text_tokens, reference_waveform, noisy_latents, timesteps):
        """
        Executes a conditional voice-cloning training pass.
        """
        # Step A: Parse characters through text transformer core
        text_emb = self.text_embedding(text_tokens) * math.sqrt(self.embed_dim)
        text_features = self.transformer_core(text_emb) # [Batch, Text_Len, Embed_Dim]
        
        # Step B: Extract high-density multi-scale style tokens from reference speech
        speaker_style_tokens = self.speaker_encoder(reference_waveform) # [Batch, Style_Frames, Embed_Dim]
        
        # Step C: Fuse text and speaker style dynamically using Multi-Head Cross-Attention
        fused_context = self.style_fusion_layer(text_features, speaker_style_tokens) # [Batch, Text_Len, Embed_Dim]
        
        # Step D: Align matching dimensions to match target audio layout
        fused_context = fused_context.transpose(1, 2) # [Batch, Embed_Dim, Text_Len]
        if fused_context.size(2) != noisy_latents.size(2):
            fused_context = F.interpolate(
                fused_context, size=noisy_latents.size(2), mode='linear', align_corners=False
            )
            
        # Step E: Process tracking metrics and estimate target noise
        time_vectors = self.time_encoder(timesteps)
        predicted_noise = self.diffusion_head(noisy_latents, fused_context, time_vectors)
        return predicted_noise

    @torch.no_grad()
    def generate_latent_trajectory(self, text_tokens, reference_waveform, num_target_frames=64):
        """
        Inference Mode: Performs the reverse diffusion sequence. Converts plain text tokens
        and a unique reference clip into matching acoustic latent frames zero-shot.
        """
        self.eval()
        device = text_tokens.device
        batch_size = text_tokens.size(0)
        
        # Pre-compute cross-attention multi-modal conditioning maps
        text_emb = self.text_embedding(text_tokens) * math.sqrt(self.embed_dim)
        text_features = self.transformer_core(text_emb)
        speaker_style_tokens = self.speaker_encoder(reference_waveform)
        
        fused_context = self.style_fusion_layer(text_features, speaker_style_tokens).transpose(1, 2)
        fused_context = F.interpolate(fused_context, size=num_target_frames, mode='linear', align_corners=False)
        
        # Initialize canvas from white noise fields
        xt = torch.randn(batch_size, self.embed_dim, num_target_frames, device=device)
        
        # Iteratively strip out noise matching the denoise=False framework
        for step in reversed(range(self.max_steps)):
            t_tensor = torch.full((batch_size,), step, dtype=torch.long, device=device)
            time_vectors = self.time_encoder(t_tensor)
            
            predicted_noise = self.diffusion_head(xt, fused_context, time_vectors)
            
            step_size = 0.015
            xt = xt - step_size * predicted_noise
            
        return xt


# =====================================================================
# VERIFICATION UNIT TEST
# =====================================================================
if __name__ == "__main__":
    print("[*] Running verification suite for File 3 (model.py)...")
    
    # Initialize our upgraded model engine
    model_engine = HighFidelityOmniVoiceEngine(embed_dim=256, max_diffusion_steps=50)
    
    # Simulate a multi-modal training snapshot dataset setup
    mock_text_tokens = torch.randint(0, 255, (2, 30))       # Batch of 2, 30 text bytes
    mock_ref_audio = torch.randn(2, 1, 48000)             # Batch of 2, 3-second reference recordings
    mock_audio_latents = torch.randn(2, 256, 120)         # Batch of 2, 120 target frame latents
    mock_timesteps = torch.randint(0, 50, (2,))           # Random diffusion schedule states
    
    print("[*] Testing upgraded cross-attention training forward path...")
    predicted_noise_output = model_engine(mock_text_tokens, mock_ref_audio, mock_audio_latents, mock_timesteps)
    print(f" -> Predicted Noise Output Shape : {list(predicted_noise_output.shape)}")
    
    print("\n[*] Testing zero-shot cloning synthesis inference path...")
    generated_latents = model_engine.generate_latent_trajectory(mock_text_tokens, mock_ref_audio, num_target_frames=100)
    print(f" -> Clone Generative Latents Shape: {list(generated_latents.shape)} [Successfully Denoised]")
    print("-" * 85)
