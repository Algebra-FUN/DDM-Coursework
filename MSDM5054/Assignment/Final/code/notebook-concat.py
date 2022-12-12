import nbformat

names = ['P3','P4','P5']

# Reading the notebooks
first_notebook = nbformat.read('P12.ipynb', 4)
# Creating a new notebook
final_notebook = nbformat.v4.new_notebook(metadata=first_notebook.metadata)
final_notebook.cells = first_notebook.cells

for name in names:
    target = nbformat.read(f'{name}.ipynb',4)
    # Concatenating the notebooks
    final_notebook.cells += target.cells

# Saving the new notebook 
nbformat.write(final_notebook, 'final.ipynb')