from stl import mesh
import numpy as np

fname = 'poster/mesh.stl'
mesh_stl = mesh.Mesh.from_file(fname)

for i in range(len(mesh_stl.vectors)):
    for j in range(3):
        px = float(mesh_stl.vectors[i][j][0])
        py = float(mesh_stl.vectors[i][j][1])
        pz = float(mesh_stl.vectors[i][j][2])

        mesh_stl.vectors[i][j][0] = pz
        mesh_stl.vectors[i][j][1] = -px
        mesh_stl.vectors[i][j][2] = py

# Write the mesh to file "mesh.stl"
mesh_stl.save('poster/mesh2.stl') 