# AI Detector UA

Desktop application for quick text analysis and basic AI-writing suspicion scoring.

The app is built with `PySide6` and provides a styled GUI for checking text, comparing it against reference samples, and exporting a plain-text report.

## Features

- Analyze text with a heuristic scoring model based on stylistic and statistical signals.
- Show a short suspicion summary and a detailed report inside the app.
- Load `.txt` and `.docx` files for analysis.
- Add baseline texts to compare writing style against reference samples.
- Save the generated report as a `.txt` file.
- Use keyboard shortcuts such as `Ctrl+A`, `Ctrl+C`, `Ctrl+V`, and `Ctrl+S`.
- Run as a Python script or package it as a standalone Windows `.exe`.

## Requirements

- Python 3.10+ recommended
- Windows is the primary target environment
- Python packages:
  - `PySide6`
  - `langdetect`
  - `razdel`
  - `python-docx`

Notes:

- `PySide6` is required to launch the GUI.
- `langdetect`, `razdel`, and `python-docx` are optional in code, but related features become limited if they are missing.
- `.docx` loading requires `python-docx`.

## Install

Clone the repository and install the dependencies:

```bash
pip install PySide6 langdetect razdel python-docx
```

Optional build dependency:

```bash
pip install pyinstaller
```

## Run

Start the application with Python:

```bash
python ai_detector.py
```

If your project uses a virtual environment, activate it first and then run the same command.

## Build

Create a standalone Windows executable with `PyInstaller`:

```bash
pyinstaller --noconfirm --clean --onefile --windowed --name ai_detector ai_detector.py
```

After a successful build, the executable will be available in the `dist/` folder.

## Usage

1. Launch the app.
2. Paste text into the main editor or open a `.txt` / `.docx` file.
3. Optionally load baseline texts for style comparison.
4. Click `Check`.
5. Review the suspicion score, key metrics, and detailed report.
6. Save the report if needed.

## Project Structure

- `ai_detector.py`: main GUI application and analysis logic

## License

Add your preferred license here before publishing on GitHub.
