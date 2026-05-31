# Word Link Bot — fully automated with a custom CNN

Built this automation for the **Word Link** game on [triumph.gg](https://triumph.gg). It reads the board, solves it, and swipes words all on its own — no human input needed once it's running.

## What it does automatically

1. Takes a screenshot of the game through **WebDriverAgent**
2. Uses a **TensorFlow CNN** I trained to recognize all 16 letters at once from the tiles
3. Runs a **DFS solver** against a curated dictionary (~75K words) to find every valid word on the board — automatically falls back to the full ENABLE word list if nothing is found
4. Physically **swipes the word path** on the screen via WDA touch actions
5. Repeats on the next board until you stop it

The whole pipeline is one continuous automation loop. No clicking, no typing, no interaction.

## Quick start

```
pip install pillow numpy requests opencv-python tensorflow
```

Calibrate tile positions for your screen:
```
python auto_calibrate.py
```

Then run:
```
python main.py
```

Optional: `--debug` dumps the preprocessed tiles + confidence scores so you can see what the CNN sees.

## Files

- `main.py` — the automation loop
- `app.py` — generates template images for rare letters (J, Q, X, Z)
- `auto_calibrate.py` — click-to-calibrate tool for tile coordinates
- `debug_ocr.py` — debug OCR on saved screenshots
- `Dictionary-curated.txt` — curated word list
- `perfect_ocr_model.h5` — the trained CNN model

---


