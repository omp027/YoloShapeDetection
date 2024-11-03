import blenderproc as bpc
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('scene',help="Path to the .obj file")
parser.add_argument('dir_name',help="Name of output directory for annotations")
parser.add_argument('output_dir',nargs='?',default="Annotation",help="Path to Output directory")
args = parser.parse_args()

bpc.init()

objs = bpc.loader.load_blend(args.scene)

for j, obj in enumerate(objs):
    obj.set_cp("category_id", j + 1)

light = bpc.types.Light()
light.set_type("POINT")
light.set_location([5,5,5])
light.set_location([5,-5,5])
light.set_energy(1000)

poi = bpc.object.compute_poi(objs)

for i in range(20):
    location = np.random.uniform([-10,-10,8],[10,10,12])
    rotation_matrix = bpc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854,0.7854))
    cam2world_matrix = bpc.math.build_transformation_mat(location,rotation_matrix)
    bpc.camera.add_camera_pose(cam2world_matrix)

bpc.renderer.enable_normals_output()
bpc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# render the whole pipeline
data = bpc.renderer.render()

# Write data to coco file
bpc.writer.write_coco_annotations(os.path.join(args.output_dir, args.dir_name),
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG")