import nbformat
nb = nbformat.read("/robodata/smodak/repos/f3rm/f3rm/demo_f3rm_pointcloud_tools.ipynb", as_version=4)
for c in nb.cells:
    if c.cell_type == "raw":
        c.cell_type = "markdown"
nbformat.write(nb, "/robodata/smodak/repos/f3rm/f3rm/new_demo_f3rm_pointcloud_tools.ipynb")
