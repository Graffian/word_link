from main import load_fallback_dictionary
strict, _, loose, _, full, _ = load_fallback_dictionary()
print(len(strict), len(loose), len(full))