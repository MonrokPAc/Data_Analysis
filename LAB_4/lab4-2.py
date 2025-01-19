from transformers import pipeline

# Завантажуємо модель через Pipeline
translator = pipeline("translation_en_to_uk", model="Helsinki-NLP/opus-mt-en-uk")

# Текст для перекладу
input_text = "Hugging Face is an amazing library for natural language processing."

# Переклад через Pipeline
result = translator(input_text, max_length=100)

# Виведення результату
print("Input:", input_text)
print("Translated:", result[0]['translation_text'])