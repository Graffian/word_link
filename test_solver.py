from main import load_dictionary, solve_board

words, prefixes = load_dictionary("Dictionary-curated.txt")
letters = ['N','D','T','D','M','I','L','K','W','K','H','H','V','K','W','N']
found = solve_board(letters, words, prefixes)
print(found)