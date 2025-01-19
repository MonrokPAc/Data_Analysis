import time
from transformers import pipeline

# Завантажуємо модель через Pipeline
translator = pipeline("translation_en_to_uk", model="Helsinki-NLP/opus-mt-en-uk")

# Функція для тестування точності перекладу
def test_translation_accuracy():
    print("=== Точність перекладу ===")
    texts = [
        "The cat is on the roof.",
        "Artificial intelligence is transforming the world.",
        "Kyiv is the capital of Ukraine.",
    ]
    for text in texts:
        result = translator(text, max_length=100)
        print(f"Input: {text}")
        print(f"Translated: {result[0]['translation_text']}\n")

# Функція для тестування стійкості до помилок
def test_resilience_to_errors():
    print("=== Стійкість до текстів із помилками ===")
    texts = [
        "Ths sentnce hs typos.",
        "U gonna lve this translater, it's awsm!",
        "Lets se hw this handels bad grammr.",
    ]
    for text in texts:
        result = translator(text, max_length=100)
        print(f"Input: {text}")
        print(f"Translated: {result[0]['translation_text']}\n")

# Функція для тестування швидкості роботи
def test_translation_speed():
    print("=== Швидкість роботи ===")
    texts = [
        "Short text.",
        "This is a medium-length sentence for testing translation speed.",
        "This paragraph contains multiple sentences. It is designed to test how long it takes for the model to translate a relatively long text."
    ]
    for text in texts:
        start_time = time.time()
        result = translator(text, max_length=200)
        end_time = time.time()
        print(f"Input: {text}")
        print(f"Translated: {result[0]['translation_text']}")
        print(f"Time taken: {end_time - start_time:.2f} seconds\n")

# Виконання тестів
if __name__ == "__main__":
    print("Тести продуктивності моделі перекладу:\n")
    test_translation_accuracy()
    test_resilience_to_errors()
    test_translation_speed()
