import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import StableDiffusionPipeline

# Перевірка наявності GPU або використання CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Завантаження CIFAR-100
def load_cifar100(batch_size=64):
    """Завантажує CIFAR-100 і повертає DataLoader."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # Перетворення у формат Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормалізація
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

# Ініціалізація DataLoader
batch_size = 64
train_loader = load_cifar100(batch_size=batch_size)

# Ініціалізація Diffusion Model
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Генерація нових зображень
def generate_images(pipeline, num_images=4):
    """Генерує нові зображення."""
    for i in range(num_images):
        image = pipeline(prompt="A colorful CIFAR-like image", height=256, width=256).images[0]
        image.save(f"generated_image_{i + 1}.png")
        print(f"Зображення {i + 1} збережено як 'generated_image_{i + 1}.png'.")

# Генерація прикладів
print("Генерація нових зображень...")
generate_images(pipeline, num_images=4)
