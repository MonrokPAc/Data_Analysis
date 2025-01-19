import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import StableDiffusionPipeline

# Перевірка наявності GPU або використання CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Завантаження Fashion MNIST
def load_fashion_mnist(batch_size=64):
    """Завантажує Fashion MNIST і повертає DataLoader."""
    transform = transforms.Compose([
        transforms.Resize(32),  # Масштабування до 32x32 (кратне 8)
        transforms.ToTensor(),  # Перетворення у формат Tensor
        transforms.Normalize((0.5,), (0.5,))  # Нормалізація
    ])

    train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

# Ініціалізація DataLoader
batch_size = 64
train_loader = load_fashion_mnist(batch_size=batch_size)

# Ініціалізація Diffusion Model
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Генерація нових зображень
def generate_images(pipeline, num_images=4):
    """Генерує нові зображення, стилізовані під Fashion MNIST."""
    for i in range(num_images):
        image = pipeline(prompt="A black-and-white image of clothing, similar to Fashion MNIST", height=32, width=32).images[0]
        image.save(f"generated_image_{i + 1}.png")
        print(f"Зображення {i + 1} збережено як 'generated_image_{i + 1}.png'.")

# Генерація прикладів
print("Генерація нових зображень...")
generate_images(pipeline, num_images=4)
