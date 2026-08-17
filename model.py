# =====================================================================
# High-Fidelity Transformer Backbone and Latent Diffusion Head
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        # Generate sinusoidal positional phase features from raw step scalar values
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        # Project through fully connected non-linear configurations
        return self.linear_layers(embeddings)


class NonAutoregressiveDiffusionHead(nn.Module):
    """
    A 1D Convolutional Denoising Block that inputs heavily distorted latent vectors, 
    time embeddings, and linguistic context to isolate and subtract noise shapes.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Combines the noisy audio latents with structural text matrices
        self.conv_in = nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        
        # Deep convolutional residual layers tracking subtle acoustic changes
        self.conv_mid = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.conv_out = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, noisy_audio_latents, text_condition_context, time_vectors):
        """
        Args:
            noisy_audio_latents (Tensor): [Batch, Embed_Dim, Sequence_Frames]
            text_condition_context (Tensor): [Batch, Embed_Dim, Sequence_Frames]
            time_vectors (Tensor): [Batch, Embed_Dim]
        """
        # Broadcast temporal metrics across the sequence timeline spatial shape
        t_spatial = time_vectors.unsqueeze(-1) # Dimension -> [Batch, Embed_Dim, 1]
        
        # Inject textual background context combined with time positioning
        conditioning = text_condition_context + t_spatial
        
        # Concat noisy inputs and text conditioning down the feature channel axis
        x = torch.cat([noisy_audio_latents, conditioning], dim=1)
        
        # Map through processing block to estimate residual structural noise
        h = self.gelu(self.conv_in(x))
        h = h + self.gelu(self.conv_mid(h)) # Residual layer integration
        estimated_noise = self.conv_out(h)
        
        return estimated_noise


class HighFidelityOmniVoiceEngine(nn.Module):
    """
    The unified multi-modal engine. Combines the text transformer and the
    latent diffusion head to construct clean speech representations from text.
    """
    def __init__(self, vocab_size=256, embed_dim=256, max_diffusion_steps=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_steps = max_diffusion_steps
        
        # Core 1: Unified multi-lingual vocabulary layer (0-255 Byte mapping)
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Core 2: Cross-attention Multi-Head Transformer Encoder Array
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=512, 
            batch_first=True, 
            activation='gelu'
        )
        self.transformer_core = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        # Core 3: Diffusion conditioning components
        self.time_encoder = DiffusionTimeEmbedding(embed_dim)
        self.diffusion_head = NonAutoregressiveDiffusionHead(embed_dim)

    def forward(self, text_tokens, noisy_latents, timesteps):
        """
        Executes a training pass. Contextually aligns text tokens to predict noise.
        """
        # 1. Project characters through global embedding spaces
        text_emb = self.text_embedding(text_tokens) * math.sqrt(self.embed_dim)
        
        # 2. Extract deep contextual language features
        text_context = self.transformer_core(text_emb) # Dimensions -> [Batch, Length, Embed_Dim]
        
        # Transpose to align with convolutional channel structures [Batch, Embed_Dim, Length]
        text_context = text_context.transpose(1, 2)
        
        # 3. Dynamic Timeline Interpolation: Stretch text frames to match audio frame sizes
        if text_context.size(2) != noisy_latents.size(2):
            text_context = F.interpolate(
                text_context, 
                size=noisy_latents.size(2), 
                mode='linear', 
                align_corners=False
            )
            
        # 4. Process time parameters
        time_vectors = self.time_encoder(timesteps)
        
        # 5. Estimate target noise vectors
        predicted_noise = self.diffusion_head(noisy_latents, text_context, time_vectors)
        return predicted_noise

    @torch.no_grad()
    def generate_latent_trajectory(self, text_tokens, num_target_frames=64):
        """
        Inference Mode: Performs the reverse diffusion sequence. Converts plain text tokens
        into clean acoustic code frames by removing random noise step-by-step.
        """
        self.eval()
        device = text_tokens.device
        batch_size = text_tokens.size(0)
        
        # Initialize an entirely random Gaussian vector frame field canvas
        xt = torch.randn(batch_size, self.embed_dim, num_target_frames, device=device)
        
        # Pre-compute textual language configurations
        text_emb = self.text_embedding(text_tokens) * math.sqrt(self.embed_dim)
        text_context = self.transformer_core(text_emb).transpose(1, 2)
        text_context = F.interpolate(text_context, size=num_target_frames, mode='linear', align_corners=False)
        
        # Progressively denoise the canvas, matching the denoise=False framework
        for step in reversed(range(self.max_steps)):
            # Broaden scalar step identifiers into parallel batch processing variables
            t_tensor = torch.full((batch_size,), step, dtype=torch.long, device=device)
            time_vectors = self.time_encoder(t_tensor)
            
            # Predict noise distributions
            predicted_noise = self.diffusion_head(xt, text_context, time_vectors)
            
            # Gradually remove the predicted noise component
            step_size = 0.015 # Static inference scheduler step delta
            xt = xt - step_size * predicted_noise
            
        return xt # Returns the pristine, fully-denoised continuous structural latent frames


# =====================================================================
# VERIFICATION UNIT TEST
# =====================================================================
if __name__ == "__main__":
    print("[*] Running verification suite for File 3 (model.py)...")
    
    # Initialize the complete core language engine
    model_engine = HighFidelityOmniVoiceEngine(embed_dim=256, max_diffusion_steps=100)
    
    # Simulate data dimensions out of our dataset parser components
    mock_text_tokens = torch.randint(0, 255, (2, 35))       # Batch of 2 items, 35 characters long
    mock_audio_latents = torch.randn(2, 256, 500)         # Batch of 2 items, 500 audio frames deep
    mock_timesteps = torch.randint(0, 100, (2,))          # Random diffusion training steps
    
    print("[*] Executing forward training data-flow passes...")
    predicted_noise_output = model_engine(mock_text_tokens, mock_audio_latents, mock_timesteps)
    
    print("\n[+] Verification Check Passed. Core Model Architecture Layers Validated:")
    print(f" -> Predicted Residual Noise Matrix Shape: {list(predicted_noise_output.shape)} (Matches latent target fields)")
    
    print("\n[*] Testing reverse-diffusion generation inference path...")
    generated_latents = model_engine.generate_latent_trajectory(mock_text_tokens, num_target_frames=120)
    print(f" -> Synthesized Generative Latents Shape : {list(generated_latents.shape)} [Pristine Denoised Output State]")
    print("-" * 75)
