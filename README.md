# Neural Style Transfer


## Introduction

This project demonstrates Neural Style Transfer using PyTorch. Neural Style Transfer allows combining the content of one image with the style of another image. This implementation uses a pre-trained VGG19 model for feature extraction and optimization to transfer styles.


## How It Works

Neural Style Transfer involves:
- **Content Image**: The image whose content we want to keep.
- **Style Image**: The image whose style we want to apply.
- **Generated Image**: The resulting image that combines the content of the content image with the style of the style image.


## Example

- **Content Image**: Two friends playing   
 ![main_img](https://github.com/Rexlin29/PRODIGY_GA_05/assets/173392854/7899681c-2bb6-4f4c-a139-3e85f4176b80)

- **Style Image**: Sunrise artwork   
 ![sub_img](https://github.com/Rexlin29/PRODIGY_GA_05/assets/173392854/3812f6cd-ff16-46e4-9ee6-731ac3e424a7)

- **Generated Image**: Combined image of the friends playing during sunrise.   
 ![Resultant_image](https://github.com/Rexlin29/PRODIGY_GA_05/assets/173392854/d611d34c-0cbd-48e3-80a0-a6294d890a19)


## Features

- **Style Transfer**: Combines the content of one image with the style of another.
- **PyTorch Implementation**: Utilizes PyTorch for efficient computation.
- **Customization**: Easily modify styles and content images for experimentation.


## Applications

Neural Style Transfer has various applications, including:

- Creating artistic effects from photographs
- Personalizing images with different artistic styles
- Enhancing visual content for presentations and marketing


## Setup

### 1. Create Project Folder and Virtual Environment

Create a new folder for your project with any name of your choice and set up a Python virtual environment using `venv`:

```sh
# Create a new folder for your project
mkdir folder_name
cd folder_name

# Create and activate the virtual environment
python -m venv env

# Activate the virtual environment (Windows)
env\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source env/bin/activate
```

### 2. Install Required Packages

Install PyTorch, torchvision and pillow:

```sh
pip install torch torchvision pillow
```

### 3. Prepare dataset

Download the content and style image. 
- **Content image**: A photograph or image whose visual characteristics (e.g., structure and objects) are retained in the final stylized output.
- **Style image**: An image or artwork whose artistic elements (e.g., colors, textures, and patterns) are applied to the content image during the style transfer process.

Place your content image and style image in your project folder. 


## Code
Follow these steps to create a Python script and integrate the provided code:

### 1. Create a Python Script:
Open your preferred text editor or integrated development environment (IDE).
Create a new Python script file (e.g., style_transfer.py).

### 2. Add the Following Code:
Copy and paste the following Python code into your script file:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import copy

# Load the images
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    
    size = max_size if max(image.size) > max_size else max(image.size)
    
    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), 
                                         (0.229, 0.224, 0.225))])
    
    # Discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    
    return image

# Display the image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    
    return image

# Load content and style images
content = load_image("path_to_your_content_image.jpg")
style = load_image("path_to_your_style_image.jpg", shape=content.shape[-2:])

# Load VGG19 model
vgg = models.vgg19(pretrained=True).features

# Freeze model parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Define the layers to use for style and content representation
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1', 
            '10': 'conv3_1', 
            '19': 'conv4_1', 
            '21': 'conv4_2',  # content representation
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram 

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

target = content.clone().requires_grad_(True)

style_weights = {
    'conv1_1': 1.,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}

content_weight = 1  # alpha
style_weight = 1e6  # beta

optimizer = optim.Adam([target], lr=0.003)
steps = 2000  # decide how many iterations to update your image (5000)

for ii in range(1, steps+1):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        _, d, h, w = target_feature.shape
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if ii % 100 == 0:
        print('Total loss: ', total_loss.item())

# Save and display the final image
final_image = im_convert(target)
final_img = Image.fromarray((final_image * 255).astype('uint8'))
final_img.save('Resultant_image.jpg')
```

### Note:
Make sure to replace the path to your content image and style image with the paths to your images.


## Execution
To perform Neural Style Transfer and create stylized images, run the script with:   
```sh
python your_script_name.py
```


## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

- Fork the repository.
- Create a new branch (`git checkout -b my-feature`).
- Make your changes and commit them (`git commit -am 'Add feature'`).
- Push to the branch (`git push origin my-feature`).
- Create a new Pull Request.   

*Please adhere to the project's coding standards and include relevant tests with your contributions.*
