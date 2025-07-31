import torch
import torch.nn as nn


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