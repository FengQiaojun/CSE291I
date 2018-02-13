# Author: 	Qiaojun Feng
# Date:		02/10/2018



# read in the vertices and the faces from .obj files
import pymesh
import pickle
print('read teapot')
teapot_mesh = pymesh.load_mesh("teapot.obj")
print('teapot imported')
print('read violin')
violin_mesh = pymesh.load_mesh("violin_case.obj")
print('violin imported')

print('teapot num_vertices: ', teapot_mesh.num_vertices, \
	' teapot num_faces: ', teapot_mesh.num_faces, \
	' teapot num_voxels: ', teapot_mesh.num_voxels)
teapot_vertices = teapot_mesh.vertices
teapot_faces = teapot_mesh.faces
teapot_mesh.enable_connectivity()
print(teapot_mesh.get_vertex_adjacent_vertices(0))

print('violin num_vertices: ', violin_mesh.num_vertices, \
	' violin num_faces: ', violin_mesh.num_faces, \
	' violin num_voxels: ', violin_mesh.num_voxels)
violin_case_vertices = violin_mesh.vertices
violin_case_faces = violin_mesh.faces


output = open('teapot_vertices.pickle', 'wb')
pickle.dump(teapot_vertices, output)
output.close()
output = open('teapot_faces.pickle', 'wb')
pickle.dump(teapot_faces, output)
output.close()
output = open('violin_case_vertices.pickle', 'wb')
pickle.dump(violin_case_vertices, output)
output.close()
output = open('violin_case_faces.pickle', 'wb')
pickle.dump(violin_case_faces, output)
output.close()