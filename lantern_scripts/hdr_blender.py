"""
Script to render IBL renders
Usage: blender <blend file> -P hdr_blender.py -b
"""
import os
import random
from math import radians
from os import listdir
from os.path import isfile, join

import bpy
import cycles
from mathutils import Euler, Matrix, Vector

# path = 'pano-videos/results/panohdr_log/chess_room/'
path = './'

datasets = ['chess_room','old_class_spot','old_class_neon','small_class'\
            ,'stairway','cafeteria']
# # out_dirs = ['prof_kitchen/','prof_living_room/','grandma_attic/'\
# #             ,'grandma_guest_room/','chess_room/'\
# #             ,'old_class_spot/','old_class_neon/','small_class/'\
# #             ,'music_room/']
# # [f'gt/{data}/' for data in datasets] + [f'ldr/{data}/' for data in datasets]  
# # [f'blender_multi/gt/{data}/' for data in datasets] + [f'blender_multi/ldr/{data}/' for data in datasets] \
inp_dirs = ['/mnt/workspace/lantern/renders/lantern-dual-exposure_no_mask/2023-11-01_022121/'] 
            # [f'results/ft_momo/{data}/mask_pred/' for data in datasets] + \
            # [f'results/1hdr/{data}/mask_pred/' for data in datasets]
out_dirs = ['/mnt/workspace/lantern/renders/lantern-dual-exposure_no_mask/2023-11-01_022121_w_objects/'] 
            # [f'blender_multi/ft_momo/{data}/mask_pred/' for data in datasets] + \
            # [f'blender_multi/1hdr/{data}/mask_pred/' for data in datasets]

# inp_dirs = ['final_0/exr/','final_1/exr/','final_2/exr/']
# out_dirs = ['final_0_render/','final_1_render/','final_2_render/']
# out_dirs = ['old_class_spot/render_ldr_500000/render/','chess_room/render_ldr_500000/render/',
#             'cafeteria/render_ldr_500000/render/','stairway/render_ldr_500000/render/',
#             'small_class/render_ldr_500000/render/','cafeteria/render_ldr_500000/render/',\
#             'prof_living_room/render_ldr_500000/render/']
pairs = list(zip(inp_dirs,out_dirs))
scn = bpy.context.scene
scn.render.engine = 'CYCLES'
scn.render.image_settings.file_format = 'OPEN_EXR'
scn.world.use_nodes = True
# Enabf_le GPU rendering.
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if "TITAN V" == device.name:
        device.use = True
    else:
        device.use = False
# select world node tree
wd = scn.world
nt = bpy.data.worlds[wd.name].node_tree

for pair in pairs:
    inp,out = pair    
    inp = inp    
    out = os.path.join(path,out)
    os.makedirs(out,exist_ok=True)
    S = bpy.context.scene
    envFiles = [f for f in listdir(inp) if isfile(join(inp, f)) and f.endswith('.exr')]
    print(envFiles)
    i = 0
    for env_map in envFiles:
        env_file = join(inp, env_map)
        bpy.ops.image.open(filepath=env_file, directory=inp, files=[
                           {"name": env_map, "name": env_map}], show_multiview=False)
        output_name = "render_" + env_map.split('.')[0] +'.exr'
        # print(output_name)
        # bpy.ops.image.save_as(save_as_render=True, copy=True, filepath=join(out_addr, output_name), show_multiview=False, use_multiview=False)


        backNode = nt.nodes['Environment Texture']
        backNode.image = bpy.data.images.load(env_file)

        bpy.context.scene.render.filepath = os.path.join(out, output_name)
        print("Output",os.path.join(out, output_name))
        bpy.ops.render.render(write_still=True)
