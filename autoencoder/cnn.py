import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoencoder(nn.Module):
    def __init__(self, image_size=512, in_channels=3, latent_dim=512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Calculate dimensions after convolutions
        # With 4 downsampling layers (each stride=2): 512 -> 256 -> 128 -> 64 -> 32
        final_feature_map_size = image_size // (2**4)  # 512 // 16 = 32
        
        # Encoder - 4 downsampling layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),    # 512x512 -> 256x256
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # 256x256 -> 128x128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * final_feature_map_size * final_feature_map_size, latent_dim)
        )
        
        # Decoder - 4 upsampling layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * final_feature_map_size * final_feature_map_size),
            nn.Unflatten(1, (256, final_feature_map_size, final_feature_map_size)),  # 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),        # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),         # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),          # 128x128 -> 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),           # 256x256 -> 512x512
            nn.Sigmoid()  # Pixel values in [0, 1]
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z  # Return reconstruction + latent vector

    def encode(self, x):
        return self.encoder(x)  # Extract latent vector 
    

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
        print("CNNAutoencoder Model Summary")
        print("=" * 50)
        print(f"Image Size: {info['image_size']}x{info['image_size']}")
        print(f"Latent Dimension: {info['latent_dim']}")
        print(f"Total Parameters: {info['total_params']:,}")
        print(f"Trainable Parameters: {info['trainable_params']:,}")
        print(f"Non-trainable Parameters: {info['non_trainable_params']:,}")
        print("=" * 50)
        
        # Print parameter breakdown by layer
        print("\nParameter Breakdown:")
        for name, module in self.named_modules():
            if len(list(module.parameters())) > 0:
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    print(f"  {name}: {params:,} parameters")




class CNNVariationalAutoencoder(nn.Module):
    def __init__(self, image_size=512, in_channels=3, latent_dim=512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Calculate dimensions after convolutions
        final_feature_map_size = image_size // (2**4)  # 512 // 16 = 32
        
        # Encoder - 4 downsampling layers with BatchNorm and LeakyReLU
        self.encoder_features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),    # 512x512 -> 256x256
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # 256x256 -> 128x128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256 * final_feature_map_size * final_feature_map_size, latent_dim)
        self.fc_logvar = nn.Linear(256 * final_feature_map_size * final_feature_map_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 256 * final_feature_map_size * final_feature_map_size)
        self.decoder_unflatten = nn.Unflatten(1, (256, final_feature_map_size, final_feature_map_size))
        
        # Decoder - 4 upsampling layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),        # 32x32 -> 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),         # 64x64 -> 128x128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),          # 128x128 -> 256x256
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # 256x256 -> 512x512
            nn.Sigmoid()  # Pixel values in [0, 1]
        )
        
        # Optional: Layer normalization for better latent space
        self.latent_norm = nn.LayerNorm(latent_dim)
    
    def encode(self, x):
        h = self.encoder_features(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # Optional: normalize latent vectors for retrieval
        z = self.latent_norm(z)
        z = self.decoder_input(z)
        z = self.decoder_unflatten(z)
        recon = self.decoder(z)
        return recon, z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, _ = self.decode(z)
        return recon, mu, logvar, z  # Return reconstruction, mu, logvar, and latent vector
    
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
        print("CNNVariationalAutoencoder Model Summary")
        print("=" * 50)
        print(f"Image Size: {info['image_size']}x{info['image_size']}")
        print(f"Latent Dimension: {info['latent_dim']}")
        print(f"Total Parameters: {info['total_params']:,}")
        print(f"Trainable Parameters: {info['trainable_params']:,}")
        print(f"Non-trainable Parameters: {info['non_trainable_params']:,}")
        print("=" * 50)
        
        # Print parameter breakdown by layer
        print("\nParameter Breakdown:")
        for name, module in self.named_modules():
            if len(list(module.parameters())) > 0:
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    print(f"  {name}: {params:,} parameters")

# VAE Loss Function
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss combining reconstruction loss and KL divergence
    beta: weight for KL divergence (beta-VAE)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

# Optional: Function to get normalized embeddings for retrieval
def get_latent_embedding(model, x, normalize=True):
    """
    Get normalized latent embeddings for image retrieval
    """
    mu, _ = model.encode(x)
    if normalize:
        mu = F.normalize(mu, p=2, dim=1)  # L2 normalization
    return mu