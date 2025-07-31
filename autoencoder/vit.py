import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
import numpy as np

class ViTAutoencoder(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, latent_dim=256, 
                 dim=384, depth=6, heads=6, mlp_dim=512, dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, dim)
        
        # Positional embedding - create for maximum possible patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer encoder
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        
        # Latent projection
        self.to_latent = nn.Linear(dim, latent_dim)
        
        # Decoder components
        self.decoder_projection = nn.Linear(latent_dim, dim)
        self.decoder_pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.decoder_transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.decoder_patch_projection = nn.Linear(dim, patch_dim)
        
        # Final reconstruction
        self.reconstruction = nn.Sigmoid()

    def forward(self, img):
        # Patch embedding
        patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                           p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embedding(patches)
        
        # Add positional embedding - make sure dimensions match
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Ensure positional embedding matches the sequence length
        if x.size(1) <= self.pos_embedding.size(1):
            x += self.pos_embedding[:, :x.size(1)]
        else:
            # If we have more patches than expected, we need to handle this
            raise ValueError(f"Input has more patches ({x.size(1)}) than expected ({self.pos_embedding.size(1)})")
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Extract class token and project to latent space
        latent = self.to_latent(x[:, 0])
        
        # Decoding
        # Expand latent to patch dimension
        decoded_patches = self.decoder_projection(latent).unsqueeze(1)
        decoded_patches = decoded_patches.repeat(1, n, 1)  # Repeat for each patch position
        
        # Ensure decoder positional embedding matches
        if decoded_patches.size(1) <= self.decoder_pos_embedding.size(1):
            decoded_patches += self.decoder_pos_embedding[:, :decoded_patches.size(1)]
        else:
            raise ValueError(f"Too many patches for decoder positional embedding")
        
        # Transformer decoding
        decoded = self.decoder_transformer(decoded_patches)
        
        # Project back to patch space
        decoded = self.decoder_patch_projection(decoded)
        
        # Reshape to image
        reconstructed = rearrange(decoded, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                h=self.image_size//self.patch_size, 
                                w=self.image_size//self.patch_size,
                                p1=self.patch_size, 
                                p2=self.patch_size)
        
        reconstructed = self.reconstruction(reconstructed)
        
        return reconstructed, latent

    def encode(self, x):
        # Patch embedding
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embedding(patches)
        
        # Add positional embedding
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Ensure positional embedding matches
        if x.size(1) <= self.pos_embedding.size(1):
            x += self.pos_embedding[:, :x.size(1)]
        else:
            raise ValueError(f"Input has more patches ({x.size(1)}) than expected ({self.pos_embedding.size(1)})")
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Extract class token and project to latent space
        latent = self.to_latent(x[:, 0])
        return latent
    
    def get_num_params(self, trainable_only=True):
        """
        Get the number of parameters in the model
        
        Args:
            trainable_only (bool): If True, count only trainable parameters
            
        Returns:
            int: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self):
        """
        Get detailed information about the model
        
        Returns:
            dict: Dictionary containing model information
        """
        total_params = self.get_num_params(trainable_only=False)
        trainable_params = self.get_num_params(trainable_only=True)
        non_trainable_params = total_params - trainable_params
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'image_size': self.image_size,
            'latent_dim': self.latent_dim
        }
    
    def print_model_summary(self):
        """
        Print a detailed summary of the model
        """
        info = self.get_model_info()
        print("=" * 50)
        print("ViT Autoencoder Model Summary")
        print("=" * 50)
        print(f"Image Size: {info['image_size']}x{info['image_size']}")
        print(f"Latent Dimension: {info['latent_dim']}")
        print(f"Total Parameters: {info['total_params']:,}")
        print(f"Trainable Parameters: {info['trainable_params']:,}")
        print(f"Non-trainable Parameters: {info['non_trainable_params']:,}")
        print("=" * 50)
        
        # Print parameter breakdown by layer
        # print("\nParameter Breakdown:")
        # for name, module in self.named_modules():
        #     if len(list(module.parameters())) > 0:
        #         params = sum(p.numel() for p in module.parameters())
        #         if params > 0:
        #             print(f"  {name}: {params:,} parameters")
    
    # Get model information
    # def get_model_info(self):
    #     """Get detailed information about the model"""
    #     total_params = sum(p.numel() for p in self.parameters())
    #     trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    #     print("=" * 50)
    #     print("ViT Autoencoder Model Summary")
    #     print("=" * 50)
    #     print(f"Image Size: {self.image_size}x{self.image_size}")
    #     print(f"Input Channels: {self.in_channels}")
    #     print(f"Patch Size: {self.patch_size}x{self.patch_size}")
    #     print(f"Patches: {(self.image_size//self.patch_size)**2}")
    #     print(f"Latent Dimension: {self.latent_dim}")
    #     print(f"Transformer Dim: {self.patch_embedding.out_features}")
    #     print(f"Total Parameters: {total_params:,}")
    #     print(f"Trainable Parameters: {trainable_params:,}")
    #     print("=" * 50)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn_norm, ff_norm in self.layers:
            attn_layer = attn_norm.fn
            residual = x
            x, _ = attn_layer(x, x, x)
            x = x + residual
            
            residual = x
            x = ff_norm(x)
            x = x + residual
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# Training function
def train_vit_autoencoder(model, train_loader, num_epochs=20, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return losses





