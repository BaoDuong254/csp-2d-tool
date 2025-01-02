# Cutting Stock Problem 2D

## Introduction
This project addresses the 2D Cutting Stock Problem, which aims to optimize the cutting of large material sheets into smaller pieces with specified dimensions, minimizing waste material.

---

## Installation and Usage Guide

### System Requirements
- Python 3.8 or higher  
- pip (Python package installer)

---

### Installation Steps

1. **Clone or download the project**:
   ```bash
   git clone <repository_url>
   cd cutting-stock-problem-2d
   ```

2. **Install required libraries**:  
   Use the following command to install all libraries listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create an executable file**:  
   Use PyInstaller to create an executable from the Python file:
   ```bash
   pyinstaller --onefile --windowed --icon=icon.ico main.py
   ```

   This command will:
   - Generate a single executable file (`--onefile`).
   - Run in windowed mode without a console (`--windowed`).
   - Use the specified icon file (`icon.ico`).

---

### Running the Program

After executing the PyInstaller command, the executable file will be located in the `dist` directory. You can run the program by either:
- Double-clicking the executable file.
- Running it through the terminal.