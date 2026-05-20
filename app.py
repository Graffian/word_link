from PIL import Image, ImageDraw, ImageFont
import os

# Create templates folder if it doesn't exist
os.makedirs("templates", exist_ok=True)

# The rare letters we need to forge
missing_letters = ["J", "Q", "X", "Z"]

for letter in missing_letters:
    # Create a perfectly white 640x640 canvas
    img = Image.new('L', (640, 640), color=255)
    draw = ImageDraw.Draw(img)
    
    # Try to load standard bold fonts (Windows or Mac)
    try:
        font = ImageFont.truetype("arialbd.ttf", 450)
    except IOError:
        try:
            font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 450)
        except IOError:
            font = ImageFont.load_default()
            print("Warning: Using default font, might be small.")
    
    # Draw the black letter perfectly in the middle
    draw.text((320, 320), letter, fill=0, anchor="mm", font=font)
    
    # Save it exactly where the bot expects it
    img.save(f"templates/{letter}.png")
    print(f"Successfully forged: {letter}.png")