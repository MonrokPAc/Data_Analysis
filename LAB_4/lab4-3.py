import torch
from diffusers import StableDiffusionPipeline

# Перевірка наявності GPU або використання CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Завантаження моделі Stable Diffusion
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, low_cpu_mem_usage=True  # Оптимізація для використання CPU
)

# Переконайтеся, що модель працює на CPU
pipeline = pipeline.to("cpu")

# Генерація зображень
def generate_image(prompt, num_images=1):
    """
    Генерує зображення на основі текстового опису (prompt).

    Args:
        prompt (str): Текстовий опис.
        num_images (int): Кількість зображень для генерації.
    """
    print(f"Генерація для запиту: '{prompt}'")
    images = pipeline(prompt, num_images_per_prompt=num_images, guidance_scale=7.5).images
    for i, img in enumerate(images):
        img.save(f"generated_image_{i + 1}.png")
        print(f"Зображення {i + 1} збережено як 'generated_image_{i + 1}.png'.")

# Використання
if __name__ == "__main__":
    # Опис, на основі якого створюється зображення
    prompt = "A duel between two different knights"
    generate_image(prompt, num_images=2)