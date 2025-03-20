import nbformat
import re

def extract_todo_patterns(notebook_path):
    # Load the notebook
    with open(notebook_path, "r") as f:
        nb = nbformat.read(f, as_version=4)
    
    # Define the regex pattern for TODOs
    pattern = re.compile(r"_# TODO: (.+?)\._")
    
    # List to store matches
    matches = []
    
    # Iterate through each cell in the notebook
    for cell in nb.cells:
        # Search for the pattern in the cell's source
        found = pattern.findall(cell.source)
        if found:
            matches.extend(found)

    return matches