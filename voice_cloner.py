# =====================================================================
# Cross-Attention Multi-Scale Acoustic Reference Style Encoder
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class InsaneVoiceClonerEncoder(nn.Module):
    """
    Extracts frame-by-frame acoustic style footprints from a target reference clip.
    Allows the model to copy vocal identities zero-shot without fine-tuning.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # A deep convolutional network that extracts speech features
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        
        # Self-Attention Layer to capture long-term speech cadences and tone rhythms
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, reference_waveform):
        """
        Args:
            reference_waveform (Tensor): Raw reference clip array [Batch, 1, Samples]
        Returns:
            Tensor: Time-aligned acoustic style tokens [Batch, Frames, Embed_Dim]
        """
        # 1. Run raw audio bits through the convolutional feature extractor
        features = self.feature_extractor(reference_waveform) # Shape: [Batch, Embed_Dim, Frames]
        
        # 2. Swap axis orientations to fit standard PyTorch Attention layouts
        features = features.transpose(1, 2) # Shape: [Batch, Frames, Embed_Dim]
        
        # 3. Apply Multi-Head Self-Attention to isolate clean voice characteristics 
        # from background microphone static or room acoustics
        attn_outputs, _ = self.self_attention(features, features, features)
        style_tokens = self.layer_norm(features + attn_outputs)
        
        return style_tokens


class CrossAttentionFusionBlock(nn.Module):
    """
    The engine component that blends linguistic sequences with acoustic voice styles.
    Replaces static vector addition to allow dynamic voice cloning.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, text_features, reference_style_tokens):
        """
        Args:
            text_features (Tensor): Base text matrices [Batch, Text_Len, Embed_Dim]
            reference_style_tokens (Tensor): Reference speaker tokens [Batch, Style_Len, Embed_Dim]
        """
        # Text query acts as the search probe, pulling matching acoustic style tones 
        # from the reference style key/value token blocks
        attn_context, _ = self.cross_attention(
            query=text_features, 
            key=reference_style_tokens, 
            value=reference_style_tokens
        )
        
        # Apply layer normalization and residual feedback connections
        x = self.layer_norm(text_features + attn_context)
        out = self.layer_norm(x + self.feed_forward(x))
        return out


# =====================================================================
# VERIFICATION UNIT TEST
# =====================================================================
if __name__ == "__main__":
    print("[*] Running verification suite for File 5 (voice_cloner.py)...")
    
    # Initialize our zero-shot style cloner engine components
    cloner_encoder = InsaneVoiceClonerEncoder(embed_dim=256)
    fusion_processor = CrossAttentionFusionBlock(embed_dim=256)
    
    # Simulate a 3-second reference voice clip at 16kHz (48,000 samples)
    mock_reference_voice = torch.randn(2, 1, 48000)
    # Simulate an encoded text sequence tensor layout
    mock_text_features = torch.randn(2, 45, 256)
    
    print(f"[*] Extracting vocal characteristics from mock reference clip...")
    extracted_style_tokens = cloner_encoder(mock_reference_voice)
    print(f" -> Style Extractor Output Shape : {list(extracted_style_tokens.shape)}")
    
    print("[*] Fusing text features with extracted speaker style...")
    fused_outputs = fusion_processor(mock_text_features, extracted_style_tokens)
    print(f" -> Fused Multi-Modal Condition Shape: {list(fused_outputs.shape)}")
    
    print("\n[+] Verification Complete. Zero-Shot Voice Cloning layers are ready.")
    print("-" * 85)
