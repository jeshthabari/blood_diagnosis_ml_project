import os

# Define folder structure
folders = [
    "data",
    "results",
    "src"
]

files = {
    "README.md": "# Blood Diagnosis Project\n\nProject initialized.",
    "src/train_cbc_model.py": "",
    "src/train_scatter_model.py": "",
    "src/evaluate_cbc_model.py": "",
    "src/evaluate_scatter_model.py": "",
    "results/.gitkeep": "",   # ensures GitHub uploads empty folder
    "data/.gitkeep": ""        # same here
}

# Create folders
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")

# Create files
for filepath, content in files.items():
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        print(f"Created file: {filepath}")

print("\nProject structure created successfully!")
