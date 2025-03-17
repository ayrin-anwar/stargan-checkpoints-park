import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Import AttGAN modules
from attgan import AttGAN
from data import check_attribute_conflict
import traceback
from collections import OrderedDict


def manipulate_face_attributes(
    image_path,
    output_path,
    model_path,
    attributes=None,
    img_size=128,
    use_gpu=True
):
    """
    Manipulate facial attributes of an image using AttGAN-PyTorch.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    output_path : str
        Path where the modified image will be saved
    model_path : str
        Path to the pretrained AttGAN model (.pth file)
    attributes : list, optional
        List of 13 attribute values (1 for add, -1 for remove, 0 for neutral) in this order:
        [Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, 
         Eyeglasses, Male, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Young]
        Default is all zeros (no change).
    img_size : int, optional
        Size to resize images to (default: 128)
    use_gpu : bool, optional
        Whether to use GPU if available (default: True)
        
    Returns:
    --------
    PIL.Image
        The modified image object
    """
    # Set default attributes if none provided
    if attributes is None:
        attributes = [0] * 13
    
    # Ensure we have exactly 13 attributes
    if len(attributes) != 13:
        raise ValueError("Exactly 13 attribute values must be provided")
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check for GPU availability
    use_cuda = use_gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # AttGAN model settings that match the original implementation
    class Args:
        def __init__(self):
            # Core attributes
            self.n_attrs = 13
            self.shortcut_layers = 1
            self.inject_layers = 0
            self.img_size = img_size
            self.enc_dim = 64
            self.dec_dim = 64
            self.dis_dim = 64
            self.dis_fc_dim = 1024
            self.enc_layers = 5
            self.dec_layers = 5
            self.dis_layers = 5
            
            # Normalization and activation functions
            self.enc_norm = 'batchnorm'
            self.dec_norm = 'batchnorm'
            self.dis_norm = 'instancenorm'
            self.dis_fc_norm = 'none'
            self.enc_acti = 'lrelu'
            self.dec_acti = 'relu'
            self.dis_acti = 'lrelu'
            self.dis_fc_acti = 'relu'
            
            # Loss weights
            self.lambda_1 = 100
            self.lambda_2 = 10
            self.lambda_3 = 1
            self.lambda_gp = 10
            
            # Training parameters
            self.mode = 'wgan'
            self.lr = 0.0002
            self.beta1 = 0.5
            self.beta2 = 0.999
            self.betas = (self.beta1, self.beta2)  # Important: required by the model
            
            # Hardware settings
            self.gpu = use_cuda
            self.multi_gpu = False
            
            # Other settings
            self.thres_int = 0.5
            self.test_int = 1.0
            self.n_sample = 1
            
        def __contains__(self, key):
            return hasattr(self, key)
    
    args = Args()
    
    # Initialize the AttGAN model
    attgan = AttGAN(args)
    
    # Load the model weights - using the specific loading method from the original repo
    print(f"Loading model from {model_path}...")
    att_id = {}
    for i, att_name in enumerate(["Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", 
                                  "Eyeglasses", "Male", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Young"]):
        att_id[att_name] = i
    
    # Modified model loading section with multiple fallbacks
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if the weights are nested under 'G' key
        if isinstance(checkpoint, dict) and 'G' in checkpoint:
            print("Found 'G' key in checkpoint, loading Generator weights...")
            attgan.G.load_state_dict(checkpoint['G'])
        else:
            attgan.G.load_state_dict(checkpoint)
            
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create a new state dictionary for adaptation
            
            new_state_dict = OrderedDict()
            
            # Check if it's the full model with nested G
            if isinstance(checkpoint, dict):
                if 'G' in checkpoint:
                    # Extract just the Generator part
                    print("Adapting state_dict with 'G' key...")
                    generator_state = checkpoint['G']
                    attgan.G.load_state_dict(generator_state)
                    print("Model loaded successfully by extracting 'G' key")
                else:
                    # Try to adapt the dictionary structure
                    print("Adapting state_dict structure...")
                    for k, v in checkpoint.items():
                        if k.startswith('module.'):
                            # Remove 'module.' prefix if using DataParallel
                            name = k[7:]  # remove 'module.'
                        else:
                            name = k
                        new_state_dict[name] = v
                    
                    attgan.G.load_state_dict(new_state_dict)
                    print("Model loaded successfully with structure adaptation")
            else:
                # If checkpoint is not a dict, create proper dict structure
                print("Creating state_dict for non-dict checkpoint...")
                attgan.G.load_state_dict({'G': checkpoint})
                print("Model loaded with non-dict adaptation")
                
        except Exception as e2:
            print(f"Alternative loading failed: {e2}")
            
            # Final attempt - modify model to accept the weights
            try:
                print("Final attempt: Modifying model structure...")
                # Print the structure of the checkpoint for debugging
                if isinstance(checkpoint, dict):
                    print("Checkpoint keys:", list(checkpoint.keys()))
                    
                    # If checkpoint contains a state_dict key
                    if 'state_dict' in checkpoint:
                        attgan.G.load_state_dict(checkpoint['state_dict'])
                        print("Model loaded using 'state_dict' key")
                    elif len(checkpoint) == 1:
                        # Try the first key if there's only one
                        first_key = list(checkpoint.keys())[0]
                        attgan.G.load_state_dict(checkpoint[first_key])
                        print(f"Model loaded using first key: {first_key}")
                    else:
                        # If all else fails, try to create a compatible model
                        print("Creating compatible model structure...")
                        
                        # Extract available layer parameters from the checkpoint
                        if 'G' in checkpoint and isinstance(checkpoint['G'], dict):
                            # Create a custom loader to match layers by name patterns
                            custom_state_dict = {}
                            for name, param in attgan.G.named_parameters():
                                for ck, cv in checkpoint['G'].items():
                                    if ck in name or name in ck:
                                        custom_state_dict[name] = cv
                                        break
                            
                            # Load partial state dict
                            attgan.G.load_state_dict(custom_state_dict, strict=False)
                            print("Loaded partial state dict with pattern matching")
                        else:
                            raise ValueError("Unable to adapt model structure to match weights")
                else:
                    raise ValueError("Checkpoint is not a dictionary structure")
                    
            except Exception as e3:
                print(f"Final loading attempt failed: {e3}")
                raise RuntimeError(f"Could not load the model. Please check if the model path is correct and matches the expected structure: {model_path}")
    
    # Set model to evaluation mode
    attgan.G.eval()
    attgan.G.to(device)
    
    # Load and preprocess the image
    print(f"Loading image from {image_path}...")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Check for attribute conflicts (as in the original implementation)
    att_names = ["Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", 
             "Bushy_Eyebrows", "Eyeglasses", "Male", "Mouth_Slightly_Open", 
             "Mustache", "No_Beard", "Pale_Skin", "Young"]
    attrs = torch.tensor(attributes).float().unsqueeze(0).to(device)

    
    # Generate the modified image
    print("Generating modified image...")
    with torch.no_grad():
        try:
            imgs_fake = attgan.G(img_tensor, attrs)
            output_tensor = imgs_fake.cpu().detach().squeeze(0)
            output_tensor = (output_tensor + 1) / 2  # Denormalize to [0, 1]
            output_img = transforms.ToPILImage()(output_tensor)
            output_img.save(output_path)
            print(f"Modified image saved to {output_path}")
            return output_img
        except Exception as e:
            print(f"Error during inference: {e}")
            print("Stack trace:")
            
            traceback.print_exc()
            raise

# Set your paths
image_path = "/kaggle/input/celebahq/CelebAHQ/CelebAHQ/Img/hq/data128x128/00111.jpg"  # Replace with your image path
output_path = "/kaggle/working/output.jpg"
model_path = "/kaggle/working/AttGAN-PyTorch/output/128_shortcut1_inject0_none/checkpoint/weights.49.pth"  # Path to model

# Create attribute list (0 = no change, 1 = add attribute, -1 = remove attribute)

# 0 ->    Bald, 
# 1 ->    Bangs, 
# 2 ->    Black_Hair, 
# 3 ->    Blond_Hair, 
# 4 ->    Brown_Hair, 
# 5 ->    Bushy_Eyebrows, 
# 6 ->    Eyeglasses, 
# 7 ->    Male, 
# 8 ->    Mouth_Slightly_Open, 
# 9 ->    Mustache, 
# 10 ->   No_Beard, 
# 11 ->   Pale_Skin, 
# 12 ->   Young

attributes = [0] * 13

# Example: Add eyeglasses and a smile
attributes[6] = 1  # Eyeglasses
# attributes[8] = 1  # Mouth_Slightly_Open (smile)

# Run the function
modified_image = manipulate_face_attributes(
    image_path=image_path,
    output_path=output_path,
    model_path=model_path,
    attributes=attributes,
    use_gpu=True
)
