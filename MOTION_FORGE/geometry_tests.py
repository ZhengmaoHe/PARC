import sys
sys.path.insert(1, sys.path[0] + ("/../.."))

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import MOTION_FORGE.polyscope_util as ps_util
import util.geom_util as geom_util
import util.torch_util as torch_util
import trimesh
import abc
from collections import OrderedDict


ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps_util.create_origin_axis_mesh()

class PS_Shape:

    def __init__(self, name, **kwargs):
        self._name = name

        if "color" in kwargs:
            self._ps_mesh.set_color(kwargs["color"])

        if "transparency" in kwargs:
            self._ps_mesh.set_transparency(kwargs["transparency"])

        transform = self._ps_mesh.get_transform()
        self._center = transform[:3, 3]

        self._ps_axes_mesh = ps_util.create_orthogonal_axes_mesh(name + " local frame")
        self._ps_axes_mesh.set_transform(transform)

        correct_init = isinstance(self._ps_mesh, ps.SurfaceMesh)
        
        assert correct_init
        if not correct_init: # weird thing to get python hints working
            self._ps_mesh = ps.register_surface_mesh("test", [0], [0])
        return
    
    def set_transform(self, transform):
        self._ps_mesh.set_transform(transform)
        self._ps_axes_mesh.set_transform(transform)
        return
    
    def get_transform(self):
        return self._ps_mesh.get_transform()
    
    def translate(self, translation):
        transform = self._ps_mesh.get_transform()
        transform[:3, 3] += translation
        self._ps_mesh.set_transform(transform)
        self._ps_axes_mesh.set_transform(transform)
        return
    
    def rotate(self, rot_mat):
        transform = self._ps_mesh.get_transform()
        transform[:3, :3] = np.matmul(rot_mat, transform[:3, :3])
        self._ps_mesh.set_transform(transform)
        self._ps_axes_mesh.set_transform(transform)
        return
    
    def get_position(self):
        transform = self._ps_mesh.get_transform()
        pos = transform[:3, 3]
        return pos
    
    def get_rotation(self):
        transform = self._ps_mesh.get_transform()
        rot = transform[:3, :3]
        return rot

    def gui(self):
        enabled = self._ps_mesh.is_enabled()
        changed, enabled = psim.Checkbox("visible", enabled)
        if changed:
            self._ps_mesh.set_enabled(enabled)

        transparency = self._ps_mesh.get_transparency()
        changed, transparency = psim.SliderFloat("transparency", transparency, v_min=0.0, v_max=1.0)
        if changed:
            self._ps_mesh.set_transparency(transparency)

        if psim.TreeNode("Transform"):
            transform = self._ps_mesh.get_transform()
            any_row_changed = False
            for i in range(4):
                row = transform[i, :]
                changed, row = psim.InputFloat4("row " + str(i), row)
                if changed:
                    transform[i, :] = row
                    any_row_changed = True
            
            if any_row_changed:
                self.set_transform(transform)

            
            psim.TreePop()

        if psim.TreeNode("Rotation"):

            psim.TreePop()
        return
    
    @abc.abstractmethod
    def support(self, dir):
        return

class PS_OBB(PS_Shape):
    def __init__(self, name, dims):
        if not isinstance(dims, np.ndarray):
            dims = np.array(dims)
        self._dims = dims

        box_trimesh = trimesh.primitives.Box(dims)
        self._ps_mesh = ps.register_surface_mesh(name, box_trimesh.vertices, box_trimesh.faces)
        super().__init__(name)
        return

    def gui(self):
        super().gui()
        dims_str = "dims: " + np.array2string(self._dims)
        psim.TextUnformatted(dims_str)
        return
    
    def get_dims(self):
        return self._dims
    
    def support(self, dir):
        # TODO
        return

class PS_Capsule(PS_Shape):
    def __init__(self, name, radius, height):
        self._radius = radius
        self._height = height

        capsule_trimesh = trimesh.primitives.Capsule(radius, height)
        self._ps_mesh = ps.register_surface_mesh(name, capsule_trimesh.vertices, capsule_trimesh.faces)
        super().__init__(name)
        return
    
    def gui(self):
        super().gui()
        radius_str = "radius: " + str(self._radius)
        height_str = "height: " + str(self._height)
        psim.TextUnformatted(radius_str)
        psim.TextUnformatted(height_str)
        return
    
    def support(self, dir):
        # TODO
        return
    
class PS_Sphere(PS_Shape):
    def __init__(self, name, radius, center, **kwargs):

        self._radius = radius
        sphere_trimesh = trimesh.primitives.Sphere(radius)
        self._ps_mesh = ps.register_surface_mesh(name, sphere_trimesh.vertices, sphere_trimesh.faces)

        transform = np.eye(4)
        transform[:3, 3] = center
        self._ps_mesh.set_transform(transform)

        self._ps_mesh.set_smooth_shade(True)

        super().__init__(name, **kwargs)
        return
    
    def gui(self):
        super().gui()
        radius_str = "radius: " + str(self._radius)
        psim.TextUnformatted(radius_str)

        changed, new_center = psim.InputFloat3("center", self._center)
        if changed:
            self._center = np.array(new_center)
            transform = self._ps_mesh.get_transform()
            transform[:3, 3] = self._center
            self._ps_mesh.set_transform(transform)
        return
    
    def support(self, dir):
        return dir * self._radius + self._center
    
class PS_Simplex(PS_Shape):
    def __init__(self, name, pA, pB, pC, pD, **kwargs):

        if not isinstance(pA, np.ndarray):
            pA = np.array(pA)
        if not isinstance(pB, np.ndarray):
            pB = np.array(pB)
        if not isinstance(pC, np.ndarray):
            pC = np.array(pC)
        if not isinstance(pD, np.ndarray):
            pD = np.array(pD)

        com = (pA + pB + pC + pD) / 4.0
        pA = pA - com
        pB = pB - com
        pC = pC - com
        pD = pD - com

        vertices = np.stack([pA, pB, pC, pD])
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0]
        ])
        
        self._ps_mesh = ps.register_surface_mesh(name, vertices, faces)
        
        transform = np.eye(4)
        transform[:3, 3] = com
        self._ps_mesh.set_transform(transform)

        super().__init__(name, **kwargs)
        return
    
    def support(self, dir):
        return


class GJK_Alg:
    def __init__(self, shape1: PS_Shape, shape2: PS_Shape):

        self._shape1 = shape1
        self._shape2 = shape2

        self._simplex = []
        return
    
    def initital_point(self):

        d = self._shape1._center - self._shape2._center
        
        d_norm = np.linalg.norm(d)
        assert d_norm > 0.0
        d = d / d_norm

        self._simplex.append(self._shape1.support(d) - self._shape2.support(-d))

        self._A = PS_Sphere(name="A", radius=0.1, center=self._simplex[0], color=[1.0, 0.0, 0.0])

        return



## GLOBALS ##
g_shapes = OrderedDict()
g_gjk_alg = None
# g_shapes[capsule1._name] = capsule1
# g_shapes[capsule2._name] = capsule2



def construct_initial_simplex(shape1: PS_Shape, shape2: PS_Shape):
    dir = shape2._center - shape1._center
    dir = dir / np.linalg.norm(dir)

    # construct the simplex
    pA = shape1.support(dir) - shape2.support(-dir)

    # The next direction is taken to be the opposite direction of the first support point
    dir = -pA / np.linalg.norm(pA)
    pB = shape1.support(dir) - shape2.support(-dir)


    # the next direction is found by taking the cross product of AB and AO
    next_dir = np.cross(pB-pA, -pA)
    if np.linalg.norm(next_dir) < 0.001:
        next_dir = np.cross(pB-pA, np.array([0.0, 0.0, 1.0]))
        dir = next_dir / np.linalg.norm(next_dir)
    pC = shape1.support(dir) - shape2.support(-dir)

    # the final direction is found by getting the normal vector of the triangle ABC
    dir = np.cross(pB - pA, pC - pA)
    dir = dir / np.linalg.norm(dir)
    pD = shape1.support(dir) - shape2.support(-dir)

    simplex = [pA, pB, pC, pD]
    return simplex

def vis_mk_diff(shape1: PS_Shape, shape2: PS_Shape):

    if isinstance(shape1, PS_Sphere) and isinstance(shape2, PS_Sphere):

        # center of sphere1, with radius expanded by the sum of the radii

        new_radius = shape1._radius + shape2._radius
        new_center = shape1._center - shape2._center
        mk_diff_shape = PS_Sphere("minkowski difference", 
                                  new_radius, 
                                  new_center,
                                  transparency=0.5)

        return
    else:
        assert False

def next_point_line(pA, pB):
    # pA: point you start with
    # pB: second point that was picked
 
    pAB = pB - pA
    pOB = - pB

    if np.dot(pAB, pOB) >= 0.0:
        dir = np.cross(np.cross(pAB, -pOB), pAB)
        return True, dir

    return False, None

def next_simplex(points, direction):

    return

def main_loop():
    global g_shapes, g_gjk_alg
    
    if psim.TreeNode("Shapes"):
        for key in g_shapes:
            curr_shape = g_shapes[key]

            if psim.TreeNode(curr_shape._name):
                curr_shape.gui()
                psim.TreePop()
        psim.TreePop()


    if psim.Button("Spheres"):
        sphere1 = PS_Sphere(name="sphere1", radius=1.0, center=[0.0, 0.0, 0.0],
                    transparency=0.6)
        sphere2 = PS_Sphere(name="sphere2", radius=0.5, center=[1.0, 0.0, 0.0],
                            transparency=0.6)
        g_shapes[sphere1._name] = sphere1
        g_shapes[sphere2._name] = sphere2
        
        g_gjk_alg = GJK_Alg(sphere1, sphere2)

    if psim.Button("Boxes"):

        box1 = PS_OBB(name="box1", dims=[1.0, 1.0, 1.0])
        box1.rotate(geom_util.np_euler2mat(0, 10, 0))
        box1.translate([0.0, 1.0, 0.0])
        box2 = PS_OBB(name="box2", dims=[0.5, 1.25, 0.8])
        box2.rotate(geom_util.np_euler2mat(-30, 30, 15))

        g_shapes[box1._name] = box1
        g_shapes[box2._name] = box2

    if psim.TreeNode("SAT"):
        
        if psim.Button("SAT"):
            box1 = g_shapes["box1"]
            box2 = g_shapes["box2"]

            pos1 = torch.tensor(box1.get_position())
            dims1 = torch.tensor(box1.get_dims() / 2.0) # obb alg takes halfwidths
            rot1 = torch_util.matrix_to_quat(torch.tensor(box1.get_rotation()))

            pos2 = torch.tensor(box2.get_position())
            dims2 = torch.tensor(box2.get_dims() / 2.0)
            rot2 = torch_util.matrix_to_quat(torch.tensor(box2.get_rotation()))
            # sep_checks, rs, axis_lens = geom_util.obb_obb(pos1, dims1, rot1, pos2, dims2, rot2)

            # print(sep_checks)
            # print(rs - axis_lens) # positive means not separating, negative means separating

            geom_util.obb_SAT(pos1, dims1, rot1, pos2, dims2, rot2)


            # sep checks is False, find the minimum of the tensor "rs - axis_lens"
            # then separate boxes along that separating axis

        psim.TreePop()

    if g_gjk_alg is not None and psim.TreeNode("GJK"):
    
        if psim.Button("Visualize Minkowsi Difference"):
            shape1 = g_shapes["sphere1"]
            shape2 = g_shapes["sphere2"]

            vis_mk_diff(shape1, shape2)

        if psim.Button("1. GJK Compute initial simplex point"):
            g_gjk_alg.initital_point()

        psim.TreePop()

    # if psim.Button("GJK test"):

    #     curr_simplex = construct_initial_simplex(shape1, shape2)

    #     pA, pB, pC, pD = curr_simplex[0], curr_simplex[1], curr_simplex[2], curr_simplex[3]

    #     if g_visualize_gjk:
    #         print(pA)
    #         print(pB)
    #         print(pC)
    #         #visualization of simplex
    #         ps_pA = PS_Sphere("pA", 0.1, pA, color=[1.0, 0.0, 0.0])
    #         ps_pB = PS_Sphere("pB", 0.1, pB, color=[0.0, 1.0, 0.0])
    #         ps_pC = PS_Sphere("pC", 0.1, pC, color=[0.0, 0.0, 1.0])
    #         ps_pD = PS_Sphere("pD", 0.1, pD, color=[0.0, 1.0, 1.0])
    #         #shape1_supp1 = PS_Sphere("shape1_supp1", 0.1, shape1.support(dir), color=[1.0, 0.0, 0.0])
    #         #shape2_supp1 = PS_Sphere("shape2_supp1", 0.1, shape2.support(-dir), color=[1.0, 0.0, 0.0])

    #         ps_simplex = PS_Simplex("simplex", pA, pB, pC, pD, transparency=0.5)

    #     #print(shape1.support(dir) - shape2.support(-dir))

    return

ps.set_user_callback(main_loop)
ps.show() # shows the UI, blocks until the UI exits