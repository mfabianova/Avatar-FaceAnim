import os
import sys
import ensurepip
import subprocess
import importlib

import bpy
import copy
import time
import math
import numpy as np


bl_info = {
    "name": "AnimFace",
    "author": "Miriam Fabianova",
    "version": (1, 0, 0),
    "blender": (3, 3, 1),
    "location": "View3D > Sidebar",
    "description": "Animates an avatar face",
    "category": "Animation"}


class install_dependencies_operator(bpy.types.Operator):
    """
    Operator for installing and importing all necessary packages.
    """

    bl_idname = "animface.instdepend_operator"
    bl_label = "Install dependencies"
    bl_description = ("Downloads and installs the required python packages for this add-on")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        try:
            import pip
        except ModuleNotFoundError:
            ensurepip.bootstrap()

        environ_copy = dict(os.environ)
        environ_copy["PYTHONNOUSERSITE"] = "1"

        python_exe = sys.executable

        subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', '--no-warn-script-location', 'pip'], env=environ_copy)
        
        # attempts to import packages
        try:
            globals()['cv2'] = importlib.import_module('cv2')
        except ModuleNotFoundError:
            subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'opencv-contrib-python'], env=environ_copy)

        try:
            globals()['mp'] = importlib.import_module('mediapipe')

            tmpmodule = importlib.import_module('mediapipe.framework.formats')
            globals()['landmark_pb2'] = tmpmodule.landmark_pb2
            del tmpmodule

        except ModuleNotFoundError:
            subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'mediapipe'], env=environ_copy)

        try:
            globals()['pyquaternion'] = importlib.import_module('pyquaternion')

            tmpmodule = importlib.import_module('pyquaternion')
            globals()['Quaternion'] = tmpmodule.Quaternion
            del tmpmodule
        except ModuleNotFoundError:
            subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'pyquaternion'], env=environ_copy)

        #face detection variables initialization
        global mpFaceMesh
        global faceMesh
        global mpDraw

        mpFaceMesh = mp.solutions.mediapipe.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(static_image_mode = False, max_num_faces=1, refine_landmarks = True) 
        mpDraw = mp.solutions.mediapipe.solutions.drawing_utils

        return {"FINISHED"}


class install_dependencies_preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        self.layout.operator(install_dependencies_operator.bl_idname, icon='CONSOLE', text="Install dependency packages")


preference_classes = (install_dependencies_operator, install_dependencies_preferences)

#---------------------global variables----------------------------------------------------------
# bools that determine visibility of rigify layers
faceVisible = True
primfaceVisible = True
secfaceVisible = True
headVisible = True

isRunning = False # determines if the video is running (not including calibration)
isRecording = False # determines if animation is being recorded

d1 = 0 # initial eye distance
# webcam dimensions
height = 0
width = 0 

# variables determined by user - armature and keyframe type
armt = None
kf_type = None 

# indices of necessary face features
# ordered in the same way as [name]_bones lists 
mouth_indices = [76, 179, 15, 403, 292, 271, 12, 41] 
brow_indices = [276, 282, 285, 55, 52, 46] 
jaw_indices = [152]
nose_indices = [439, 0, 219]
eye_indices = [386, 253, 159, 23]
indices = [mouth_indices, brow_indices, jaw_indices, eye_indices, nose_indices]

pose_bones = [] # list of all pose bones to be moved
head_bone = None 

old_positions = [] # global bone positions
polygon_edit = [] # polygon v priestore avatara

rec_animation = [] # list of recorded keyframes

error_message = '' 

#variables for face detection
mpFaceMesh = None
faceMesh = None
mpDraw = None



#------------------------------------------functiohs-------------------------------------------------------
def filter_callback(self, object):
    return object.name in self.my_collection.objects.keys()

def update_bool(self,context): 
    """Updates bool properties with every change."""
    bpy.data.armatures["Human.rigify"].show_names = bpy.context.scene.my_tool.show_names
    bpy.data.armatures["Human.rigify"].show_bone_custom_shapes = bpy.context.scene.my_tool.show_shapes
    bpy.data.armatures["Human.rigify"].show_group_colors = bpy.context.scene.my_tool.show_col
    bpy.data.armatures["Human.rigify"].show_axes = bpy.context.scene.my_tool.show_axes
    bpy.data.armatures["Human.rigify"].axes_position = bpy.context.scene.my_tool.axes_offset
    if bpy.context.scene.my_collection_objects is not None:
        bpy.data.objects[bpy.context.scene.my_collection_objects.name].show_in_front = bpy.context.scene.my_tool.show_front

def create_normalized_landmark_list(landmarks):
    """Returns NormalizedLandmarList out of an input array."""
    normalized_landmarks = list(map(lambda p: landmark_pb2.NormalizedLandmark(x=p[0], y=p[1], z=p[2]), landmarks))
    return landmark_pb2.NormalizedLandmarkList(landmark=normalized_landmarks)

def triangle_normal(p1, p2, p3):
    """Returns a normal of a triangle determined by points p1, p2, p3."""
    e1 = (p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2]) 
    e2 = (p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2]) 
    return np.cross(e2,e1)

def dist(p1, p2):
    """Returns Euclidean distance of two input points p1, p2."""
    if len(p1) == 3:
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    elif len(p2) == 2:
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    else:
        return
    
def normalize(v): 
    """Normalizes the input vector v."""
    s = 0 # sum of squares
    for i in range(len(v)):
        s += v[i]**2
    v_norm = v/np.sqrt(s)
    return v_norm

def calibrate(context):
    """
    Calibrates camera by finding the initial eye distance. 
    
        Returns: 
            d1 (float): Initial eye distance
    """
    global d1
    global height, width

    vid = cv2.VideoCapture(0) # starts recording
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    while(True):
        # read the next frame
        succ, frame = vid.read() 
        height, width = frame.shape[:2] 

        # displaying the frame 
        flipped = cv2.flip(frame, 1)
        cv2.putText(flipped, "Look straight into the camera", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,127,0), thickness=2, lineType=cv2.LINE_AA) 
        cv2.putText(flipped, "and press c to calibrate",      (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,127,0), thickness=2, lineType=cv2.LINE_AA) 
        cv2.rectangle(flipped, (int(width/2 - 70), int(height/2 - 100)), (int(width/2 + 70), int(height/2 + 100)), (0,255,0), 2)
        cv2.line(flipped, (int(width/2 - 10), int(height/2 - 10)), (int(width/2 + 10), int(height/2 + 10)), (0,255,0), 2)
        cv2.line(flipped, (int(width/2 - 10), int(height/2 + 10)), (int(width/2 + 10), int(height/2 - 10)), (0,255,0), 2)
        cv2.imshow("Calibration", flipped)

        # face detection
        full_lm = get_face_lm(flipped)
        
        # determining d1 after c key is pressed
        if(cv2.waitKey(1) & 0xFF == ord('c')):
            eye_centerR = ((full_lm.landmark[159].x + full_lm.landmark[145].x)*width/2, (full_lm.landmark[159].y + full_lm.landmark[145].y)*height/2, (full_lm.landmark[159].z + full_lm.landmark[145].z)*width/2) #priemer landmarkov 159 a 145
            eye_centerL = ((full_lm.landmark[386].x + full_lm.landmark[374].x)*width/2, (full_lm.landmark[386].y + full_lm.landmark[374].y)*height/2, (full_lm.landmark[386].z + full_lm.landmark[374].z)*width/2) #priemer landmarkov 386 a 374 
            d1 = dist(eye_centerL, eye_centerR)
            break
            
    vid.release()
    cv2.destroyAllWindows()

def get_angle(p1, p2, p3):
    """Returns angle <p1p2p3."""
    v0 = (p2[0] - p1[0], p2[1]-p1[1]) # vector p1p2
    v1 = (p2[0] - p3[0], p2[1] - p3[1]) # vector p3p2

    v0_norm = np.sqrt(v0[0]**2 + v0[1]**2) 
    v1_norm = np.sqrt(v1[0]**2 + v1[1]**2)

    angle = np.arccos((v0[0] * v1[0] + v0[1]*v1[1])/(v0_norm*v1_norm))
    return angle

def mean_value_bc(point, vertices): 
    """
    Computes mean value barycentric coordinates.
    
        Input:
            point (tuple): Point whose coordinates are being computed
            vertices (list): Vertices of the reference polygon

        Output:
            w (list): Barycentric oordinates of point
    """
    n = len(vertices)
    EPS = 1e-5
    
    s = n * [(0.0, 0.0)] # vectors from vertices to point
    r = n * [0.0]        # distances from vertices to point
    area2 = n * [0.0]    # area * 2
    dot_prod = n * [0.0] # dot producs
    w = n * [0.0]        # final barycentric coordinates
    
    for i in range(n):
        s[i] = (vertices[i][0] - point[0], vertices[i][1] - point[1])
        r[i] = np.sqrt(s[i][0]**2 + s[i][1]**2) 
        if abs(r[i]) < EPS: # is point near a vertex?
            w[i] = 1.0
            return w
    
    for i in range(n):
        ip1 = (i + 1) % n # next vertex index 
        area2[i] = (s[i][0]*s[ip1][1] - s[ip1][0]*s[i][1])/2
        dot_prod[i] = s[i][0]*s[ip1][0] + s[i][1]*s[ip1][1]
    
    sum_w = 0
    
    for i in range(n):
        ip1 = (i + 1) % n            # next vertex index
        im1 = ((i - 1) % n + n) % n  # previous vertex index
        if abs(area2[ip1]) > EPS:   # is point near a vertex?
            w[i] += (r[im1] - dot_prod[im1] / r[i]) / area2[im1] # compute barycentric coordinates
        if abs(area2[im1]) > EPS:   # is point near a vertex?
            w[i] += (r[ip1] - dot_prod[i] / r[i]) / area2[i]     # compute barycentric coordinates
        else: #vertex is near a polygon edge
            w = np.zeros(len(vertices))
            #compute coordinates of point as combination of edge vertices
            w[i] = r[ip1]/(r[ip1] + r[i])
            w[ip1] = r[i]/(r[ip1] + r[i]) 
            return w
        sum_w += w[i]
            
    for i in range(n):
        w[i] /= sum_w # normalization

    return w

def get_face_lm(_frame):  
    """
    Detects face landmarks in a frame.

        Input:
            _frame (ndarray): Current frame in which face is detected

        Output:
            full_lm (NormalizedLandmarkList): List of detected landmarks
    """
    rgbFrame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB) # color conversion
    
    _frame.flags.writeable = False # improve perf
    landmarks = faceMesh.process(rgbFrame) # detection
    _frame.flags.writeable = True

    if landmarks.multi_face_landmarks is None:
        return None
    
    full_lm = landmarks.multi_face_landmarks[0] 
    return full_lm 
                    

def get_mesh_normal(_landmarks):
    """
    Computes normal of an input mesh. 

    Mesh is determined by vertices in the _landmarks list. Normal of the mesh is computed as the average of chosen triangle normals.

        Input:
            _landmarks (list): List of mesh vertices

        Output:
            normal (tuple): Normal of the mesh
    """
    indices = [332, 389, 323, 397, 378, 149, 172, 93, 162, 103] # indices in the _landmark list of the vertices of chosen triangles

    neighbour_lm = [] # vertices of chosen triangles
    for i in indices:
        neighbour_lm.append((_landmarks[i][0] * width, _landmarks[i][1] * height, _landmarks[i][2] * width))
    
    triangle_normals = [] # normals of chosen triangles
    for i in range(len(neighbour_lm)):
        curr_n = triangle_normal(neighbour_lm[(i-1)%len(indices)],neighbour_lm[i], (_landmarks[1][0] * width, _landmarks[1][1] * height, _landmarks[1][2] * width))
        triangle_normals.append(curr_n)

    normal = np.mean(np.array(triangle_normals), axis=0) # average vector of the triangle normals
    return normal

def rotate_xy(_lm, _normal, _rotation_pt):
    """
    Rotates landmarks around axes x and y.

    Landmarks are rotated so that the normal of mesh they determine is equal to (0,0,1). 
    Rotation is applied via quaternions. Each landmark is rotated as vector with beginning in the nose landmark and ending in current landmark thats rotated.

        Input:
            _lm (list): List of landmarks
            _normal (tuple): Normal of the mesh that is determined by landmarks
            _rotation_pt (tuple): The center of rotation

        Output:
            rotated_lm (list): List of landmarks after rotation
            angle (float): Angle used for rotation
            rotation_vector (tuple): Rotation vector used for rotation
    """
    rotated_lm = [_lm[0]] # list of landmarks after rotation
    normalized = normalize(_normal) 
    
    rotation_vector = np.cross(normalized, (0,0,1)) # vector determining axis of rotation
    angle = np.arccos(np.dot((0,0,1), normalized)) # angle of rotation

    q =  Quaternion(axis=[rotation_vector[0], rotation_vector[1], rotation_vector[2]], angle = angle) #quaternion by which is rotation applied
    for i in range(468):
        if i != 1: # leave out nose landmark (initially index 1) to move it to the front of the list
            lm_vec = (_lm[i][0] - _rotation_pt[0],_lm[i][1] - _rotation_pt[1],_lm[i][2] - _rotation_pt[2]) # vector of the current landmark to be rotated
            rotated_vec = q.rotate(lm_vec) # rotated vector 
            rotated_lm.append((_rotation_pt[0] + rotated_vec[0], _rotation_pt[1] + rotated_vec[1], _rotation_pt[2] + rotated_vec[2])) # insert rotated landmark determined by rotated vector into the list
         
    return rotated_lm, angle, rotation_vector

def translate(_lm, _vec):
    """
    Translates landmarks by given translation vector.

        Input:
            _lm (list): List of landmarks
            _vec (tuple): Translation vector

        Output:
            translated_lm (list): List of translated landmarks
    """
    translated_lm = [] 
    for i in range(468):
        landmark = (_lm[i][0] + _vec[0], _lm[i][1] + _vec[1])
        translated_lm.append(landmark)
    return translated_lm

def scale(_lm, _fac): 
    """
    Scales landmarks by given scaling factor.

    Scaling happends around the point (0,0). Function creates a vector beginning in this point and ending in the current landmark,
    scales the vector and then extracts the new landmark position out of the resulting scaled vector. 

        Input: 
            _lm (list): List of landmarks
            _fac (float): Scaling factor

        Output:
            scaled_lm (list): List of scaled landmarks
    """
    center = _lm[0]
    scaled_lm = []
    for i in range(468):
        landmark = (_fac * (_lm[i][0] - center[0]) + center[0], _fac * (_lm[i][1] - center[1]) + center[1]) 
        scaled_lm.append(landmark)
        
    return scaled_lm

def rotate_z(_lm, _forehead):
    """
    Rotates landmarks around the z axis.

    Landmarks are rotated around the point (0,0), so that the vector beginning in this point and ending in fixed forehead landmark 
    maps to vector (0,-1).

        Input:
            _lm (list): List of landmarks
            _forehead (tuple): Forehead landmark

        Output:
            rotated_z_lm (list): List of rotated landmarks
    """
    rotated_z_lm = [(0,0)]
    forehead_angle = np.arctan((_forehead[0])/np.abs(_forehead[1])) # angle of rotation

    if _forehead[1] > 0: # flipping angle if vector aims downwards
        forehead_angle = math.pi - forehead_angle

    for i in range(1, 468):
        vec = (_lm[i][0], _lm[i][1]) # vector being rotated
        x =  math.cos(forehead_angle) * vec[0] + math.sin(forehead_angle) * vec[1]
        y = -math.sin(forehead_angle) * vec[0] + math.cos(forehead_angle) * vec[1] 
                        
        landmark = (x, y)
        rotated_z_lm.append(landmark)
    return rotated_z_lm


def bone_movement(_lm, _indices, _polygon_img, _polygon_edit, _old_positions, _pose_bones, group = None):
    """
    Moves bones according to current landmark position.

    New position of bones is calculated with the help of barycentric coordinates. Depending on the old positions
    and newly found positions difference vector is found and used to move bones.

        Input:
            _lm (list): List of landmarks 
            _indices (list): List of indices of the landmarks used to determine the position of bones
            _polygon_img (list): Vertex positions of the polygon in image space
            _polygon_edit (list): Vertex positions of the polygon in avatar space
            _old_positions (list): Positions of bones found in the last cycle
            _pose_bones (list): List of names of bones used to movement
            group (string): Indicator of group of bones thats being moved

        Output:
            future_old_positions (list): List of new positions of bones that were found
            group_diff (list): List of difference vectors that were found
    """
    future_old_positions = [] # list of new bone positions
    group_diff = [] # list of difference vectors
    for i in range(len(_indices)): #cycle through bones/indices
        coor = mean_value_bc(_lm[_indices[i]], _polygon_img) # compute coordinates

        # compute new position and save to list of new positions
        new_position = (0,0)
        for j in range(len(_polygon_edit)):
            new_position += _polygon_edit[j] * coor[j]  
        future_old_positions.append(new_position) 

        old_position = _old_positions[i] # old position of current bone
        difference = (new_position[0]-old_position[0], new_position[1]-old_position[1]) # compute difference vector of old and new positions

        # difference adjustment
        if group == 'MOUTH':
            if i in {1,3,5,7}: # secondary bones move less
                difference = (0.5 * difference[0], 0.5 * difference[1])
    
            if 0<i and i<4: # bones of bottom lip are moved with jaw as well 
                difference = (0.4 * difference[0], 0.4 * difference[1])
        elif group == 'JAW': # jaw bone has flipped y axis
            difference = (difference[0], -difference[1])
        elif group == 'EYE': 
            difference = (1.25 * difference[0], 1.25 * difference[1])
        group_diff.append(difference)
        
        _pose_bones[i].location = (_pose_bones[i].location.x + difference[0],_pose_bones[i].location.y+ difference[1], _pose_bones[i].location.z) #bone movement
                        
    return future_old_positions, group_diff

def one_euro(_lm, _isFirst, _dx_prev, _x_prev):
    """
    Smoothing of the current landmark list using one euro filter.

        Input:
            _lm (list): List of landmarks
            _isFirst (bool): Indicator of the first cycle
            _dx_prev (list): List of differences from the previous cycle
            _x_prev (list): List of landmarks from the previous cycle

        Output:
            x_hat (list): List of smooth landmarks
            dx_hat (list): List of smooth differences
    """
    # parameters
    span = 10
    f_cutoff = 0.01
    threshold =  1.25
    dx_scale = 600

    x = np.array(_lm)
    if _isFirst: #initialization if the cycle is first 
        _dx_prev = np.array(len(_lm) * [(0.0,0.0,0.0)]) 
        _x_prev = x 
    
    dx = dx_scale * (x -_x_prev) 
    
    alpha_dc = 2 / (1 + span)
    dx_hat = np.add(alpha_dc * dx, (1 - alpha_dc) * _dx_prev) 

    cutoff = f_cutoff + np.abs(dx_hat)
    alpha_fc = 2 / (1 + span / cutoff)
    x_hat = np.around(np.where(np.abs(dx) < threshold, _x_prev, alpha_fc * x + (1 - alpha_fc) * _x_prev), decimals=3)
    
    return x_hat, dx_hat

def init_bone_positions():
    """Initialization of armature and bone positions."""
    global mouth_bones, brow_bones, jaw_bones, nose_bones, eye_bones # bone name groups
    global error_message # for displaying errors

    global pose_bones 
    global head_bone 
    global old_positions 
    global polygon_edit 
    
    bpy.ops.object.mode_set(mode='EDIT')
    
    # armature initialization
    ob = bpy.data.objects[armt.name]
    armature_edit = ob.data
    
    # error message if chosen object is not armature
    try:
        editbones = armature_edit.edit_bones
    except Exception:
        error_message = 'The chosen object is not an armature.'
        return
   
    # error message if bone names are wrong
    tool = bpy.context.scene.my_tool
    string_prop = [tool.head, tool.chin, tool.mouth_r, tool.mouth_br, tool.mouth_b, tool.mouth_bl, tool.mouth_l, tool.mouth_tl, tool.mouth_t,
                   tool.brow_l1, tool.brow_l2, tool.brow_l3, tool.brow_r3, tool.brow_r2, tool.brow_r1, tool.jaw,
                   tool.nose_l, tool.nose_m, tool.eye_l, tool.eye_tl, tool.eye_bl, tool.eye_r,tool.eye_tr, tool.eye_br]
    for name in string_prop:
        bone = armature_edit.edit_bones.get(name)
        if bone is None:
            error_message = 'There is no bone with name ' + name +'.'
            return
        
    # bone name initialization
    mouth_bones = [tool.mouth_r,tool.mouth_br,tool.mouth_b,tool.mouth_bl,tool.mouth_l,tool.mouth_tl, tool.mouth_t, tool.mouth_tr]
    brow_bones = [tool.brow_l1, tool.brow_l2,tool.brow_l3, tool.brow_r3, tool.brow_r2, tool.brow_r1] 
    jaw_bones = [tool.jaw]
    nose_bones = [tool.nose_l, tool.nose_m, tool.nose_r] 
    eye_bones = [tool.eye_tl, tool.eye_bl, tool.eye_tr, tool.eye_br]

    # polygon bones initialization
    chinBone_edit = armature_edit.edit_bones[tool.chin]
    mouthBoneL_edit = armature_edit.edit_bones[tool.mouth_l]    
    noseBoneL_edit = armature_edit.edit_bones[tool.nose_l]   
    eyeBoneL_edit = armature_edit.edit_bones[tool.eye_l] 
    browBoneL3_edit = armature_edit.edit_bones[tool.brow_l3]
    browBoneR3_edit = armature_edit.edit_bones[tool.brow_r3]
    eyeBoneR_edit = armature_edit.edit_bones[tool.eye_r] 
    noseBoneR_edit = armature_edit.edit_bones[tool.nose_r]  
    mouthBoneR_edit = armature_edit.edit_bones[tool.mouth_r] 
    
    polygon_edit = np.array([(chinBone_edit.head.x, chinBone_edit.head.z),
                             (mouthBoneL_edit.head.x, mouthBoneL_edit.head.z),
                             (noseBoneL_edit.head.x, noseBoneL_edit.head.z),
                             (eyeBoneL_edit.head.x, eyeBoneL_edit.head.z),
                             (browBoneL3_edit.head.x, browBoneL3_edit.head.z),
                             (browBoneR3_edit.head.x, browBoneR3_edit.head.z),
                             (eyeBoneR_edit.head.x, eyeBoneR_edit.head.z),
                             (noseBoneR_edit.head.x, noseBoneR_edit.head.z),
                             (mouthBoneR_edit.head.x, mouthBoneR_edit.head.z)])

#-------------------bone positions initialization-----------------------
    # mouth bones positions
    mouth_old_positions = []
    for i in range(len(mouth_bones)):
        lmBone_edit = armature_edit.edit_bones[mouth_bones[i]]
        mouth_old_positions.append((lmBone_edit.head.x, lmBone_edit.head.z)) 

    # jaw bones positions
    jaw_old_positions = []
    for i in range(len(jaw_bones)):
        lmBone_edit = armature_edit.edit_bones[jaw_bones[i]]
        jaw_old_positions.append((lmBone_edit.tail.x, lmBone_edit.tail.z))

    # brow bones positions
    brow_old_positions = []
    for i in range(len(brow_bones)):
        lmBone_edit = armature_edit.edit_bones[brow_bones[i]]
        brow_old_positions.append((lmBone_edit.head.x, lmBone_edit.head.z))

    # nose bones positions
    nose_old_positions = []
    for i in range(len(nose_bones)):
        lmBone_edit = armature_edit.edit_bones[nose_bones[i]]
        nose_old_positions.append((lmBone_edit.head.x, lmBone_edit.head.z))
        
    # eye bones positions
    eye_old_positions = []
    for i in range(len(eye_bones)):
        lmBone_edit = armature_edit.edit_bones[eye_bones[i]]
        eye_old_positions.append((lmBone_edit.head.x, lmBone_edit.head.z))

    old_positions = [mouth_old_positions, brow_old_positions, jaw_old_positions, eye_old_positions, nose_old_positions]
        
#-------------------pose bones initialization----------------------------
    bpy.ops.object.mode_set(mode='POSE')
    armature_pose = bpy.context.scene.objects[armt.name] 

    # mouth pose bones
    mouth_pose_bones = []
    for i in range(len(mouth_bones)):
        lmBone_pose = armature_pose.pose.bones.get(mouth_bones[i])
        mouth_pose_bones.append(lmBone_pose)

    # jaw pose bones
    jaw_pose_bones  =[]
    for i in range(len(jaw_bones)):
        lmBone_pose = armature_pose.pose.bones.get(jaw_bones[i])
        jaw_pose_bones.append(lmBone_pose)    

    # brow pose bones
    brow_pose_bones = []
    for i in range(len(brow_bones)):
        lmBone_pose = armature_pose.pose.bones.get(brow_bones[i])
        brow_pose_bones.append(lmBone_pose)

    # nose pose bones
    nose_pose_bones = []
    for i in range(len(nose_bones)):
        lmBone_pose = armature_pose.pose.bones.get(nose_bones[i])
        nose_pose_bones.append(lmBone_pose)

    # eye pose bones
    eye_pose_bones = []
    for i in range(len(eye_bones)):
        lmBone_pose = armature_pose.pose.bones.get(eye_bones[i])
        eye_pose_bones.append(lmBone_pose)

    pose_bones = [mouth_pose_bones, brow_pose_bones, jaw_pose_bones, eye_pose_bones, nose_pose_bones]
                
    head_bone = armature_pose.pose.bones.get(bpy.context.scene.my_tool.head)

def create_anim():
    """Creates animation based on recorded positions determined by difference vectors."""
    
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.user_transforms_clear() 
    bpy.ops.pose.select_all(action='DESELECT')

    for all_diff in rec_animation: # all_diff is a list of difference vectors in one frame
        for i in range(len(all_diff)): # cycle through groups of bones - group_diff in frame 
            group_diff = all_diff[i]
            for j in range(len(group_diff)): # cycle through every bone in group 
                if i != len(all_diff)-1: # if not in the last group - face movements
                    pose_bones[i][j].location = (pose_bones[i][j].location.x + group_diff[j][0], pose_bones[i][j].location.y + group_diff[j][1], pose_bones[i][j].location.z)
                else: # if in the last group - head movement
                    head_bone.bone.select = True

                    # rotation around x and y
                    head_bone.rotation_mode = 'QUATERNION'
                    head_bone.rotation_quaternion = (group_diff[1], group_diff[2], -group_diff[3], group_diff[4])
                
                    # rotation around z
                    # ak su kosti skyte, odkryju sa rotuju a skryju spat
                    isHidden = bpy.context.active_object.hide_get()
                    bpy.context.active_object.hide_set(False)
                    bpy.ops.transform.rotate(value = group_diff[0], orient_axis='Z', orient_type= 'LOCAL')
                    if isHidden:
                        bpy.context.active_object.hide_set(True)
            
            
        # insert keyframe and move 3 frames further
        bpy.ops.pose.select_all(action='SELECT')
        isHidden = bpy.context.active_object.hide_get() 
        bpy.context.active_object.hide_set(False)
        bpy.ops.anim.keyframe_insert_by_name(type=kf_type) 
        bpy.context.scene.frame_current += 3
        if isHidden:
            bpy.context.active_object.hide_set(True)
            
        bpy.ops.pose.select_all(action='DESELECT')
        
    rec_animation.clear()
    bpy.context.scene.frame_current = 0


        
#--------------------------------------------------------------------------------------

class ModalTimerOperator(bpy.types.Operator):
    """Operator which runs itself from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"
    
    global isRunning 
    global isRecording 
    global kf_type 

    global height, width 
    global d1 

    global pose_bones 
    global head_bone 
    global old_positions

    global rec_animation

    #---------------------------------------------
    vid = None # video capture variable

    # lists used for smoothing
    x_hat_prev_face = np.array(468 * [(0.0, 0.0,0.0)]) 
    dx_hat_prev_face = np.array(468 * [(0.0,0.0,0.0)])
    
    isFirst= True #bool indicating the first frame

    counter = 0 #frame counter to collect data
    plot_points = [] #list for unsmooth data 
    smooth_plot_points = [] #list for smooth data
    
    polygon_indices = [152, 292, 439, 263, 285, 55, 33, 219, 76] #indices of landmarks that determine polygon in image space
    polygon_img = [] #position of landmarks that determine polygon in image space
    
    prevTime = 0
    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'} or isRunning == False:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # if camera has not been calibrated yet
            if width == 0:
                calibrate(context)

            # load video
            if self.vid == None:
                self.vid = cv2.VideoCapture(0)
                self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            
            succ, frame = self.vid.read() # reading frame from the video
            flipped = cv2.flip(frame,1)

            full_lm = get_face_lm(flipped) # detection
            if full_lm is None:
                cv2.imshow('CameraVideo', flipped) 
                return {'PASS_THROUGH'}
            
            smooth_lm, dx_list = one_euro(np.array([(lm.x, lm.y, lm.z) for lm in full_lm.landmark]), self.isFirst,self.dx_hat_prev_face, self.x_hat_prev_face) #smoothing

            if len(smooth_lm) != 0: # if detected
                self.x_hat_prev_face = copy.deepcopy(smooth_lm) 
                self.dx_hat_prev_face = copy.deepcopy(dx_list)

                mpDraw.draw_landmarks(flipped, create_normalized_landmark_list(smooth_lm), mpFaceMesh.FACEMESH_CONTOURS,
                                 mpDraw.DrawingSpec(thickness = 1, circle_radius= 1, color = (0,255,0)), mpDraw.DrawingSpec(thickness = 1, color = (0,0,255)))

            # fps calculation
            currTime = time.time()
            fps = int(1/(currTime - self.prevTime))
            self.prevTime = currTime

            # frame display
            cv2.putText(flipped, "fps = " + str(fps), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), thickness = 3)
            cv2.imshow('CameraVideo', flipped) 
            
            # determining current distance of user from camera
            eye_centerR = ((smooth_lm[159][0] + smooth_lm[145][0])*width/2, (smooth_lm[159][1] + smooth_lm[145][1])*height/2, (smooth_lm[159][2] + smooth_lm[145][2])*width/2) #priemer landmarkov 159 a 145
            eye_centerL = ((smooth_lm[386][0] + smooth_lm[374][0])*width/2, (smooth_lm[386][1] + smooth_lm[374][1])*height/2, (smooth_lm[386][2] + smooth_lm[374][2])*width/2) #priemer landmarkov 386 a 374
            d = d1/dist(eye_centerL, eye_centerR)

            #variables needed for transformation into normalized form
            nose_lm = (smooth_lm[1][0] * width, smooth_lm[1][1] * height, smooth_lm[1][2] * width) #nose landmark position in video
            avg_normal = get_mesh_normal(smooth_lm) #normal of mesh
            nose_proj = (nose_lm[0], nose_lm[1]) #nose landmark projection
            translation_vec = (- nose_proj[0], - nose_proj[1]) 
            scaling_fac = d  

            #landmark positions in video
            img_landmarks = []
            img_landmarks.append(nose_lm)
            for i in range(468):
                if i != 1: 
                    landmark = (smooth_lm[i][0] * width, smooth_lm[i][1] * height, smooth_lm[i][2] * width)
                    img_landmarks.append(landmark) 

            #transformation into normalized form
            rotated_landmarks, theta, rotation_vector = rotate_xy(img_landmarks, avg_normal, nose_lm)
            translated_landmarks = translate(rotated_landmarks, translation_vec)
            scaled_landmarks = scale(translated_landmarks, scaling_fac)
            forehead = scaled_landmarks[10]
            rotated_z_landmarks = rotate_z(scaled_landmarks, forehead)

            final_landmarks = rotated_z_landmarks
            
            #-------------head bone rotation-------------------
            vec = (img_landmarks[263][0] - img_landmarks[33][0], img_landmarks[263][1]-img_landmarks[33][1]) # eye vector
            alpha =  np.arctan((vec[1])/np.abs(vec[0]-1)) # angle between eye vector and vector (-1,0)
            
            # rotation around x and y axes
            head_bone.bone.select = True
            head_bone.rotation_mode = 'QUATERNION'
            head_bone.rotation_quaternion = (theta, rotation_vector[0], -rotation_vector[1], rotation_vector[2])
                
            # rotation around z axis
            isHidden = bpy.context.active_object.hide_get()
            bpy.context.active_object.hide_set(False)
            bpy.ops.transform.rotate(value = alpha, orient_axis='Z', orient_type= 'LOCAL') #rotacia
            if isHidden:
                bpy.context.active_object.hide_set(True)
            
            # polygon_img initialization
            if self.isFirst:     
                self.isFirst = False
                self.polygon_img.clear()
                for i in self.polygon_indices:
                    self.polygon_img.append(final_landmarks[i])
                
            # face bones rotation
            opt = ['MOUTH', None, 'JAW', 'EYE', None] # indicator of bone group
            all_diff = [] # list of differences for animation
            for i in range(len(indices)):
                old, diff = bone_movement(final_landmarks, indices[i], self.polygon_img, polygon_edit, old_positions[i], pose_bones[i], opt[i])
                old_positions[i] = copy.deepcopy(old)
                all_diff.append(diff)
            
            '''
            #plotting 
            if self.counter < 200:
                self.plot_points.append(final_landmarks[292])
                self.smooth_plot_points.append(smooth_landmarks[292])
            elif self.counter == 200:
                print(len(self.plot_points))
                with open("vyhladzovacie_data_no_movement.txt",'w') as file:
                    file.write(str(self.plot_points) + "\n")
                    file.write(str(self.smooth_plot_points))
            '''
            if isRecording: # record positions for animation
                    all_diff.append([alpha, theta, rotation_vector[0], rotation_vector[1], rotation_vector[2]])
                    rec_animation.append(all_diff)
                
            self.counter += 1

        return {'PASS_THROUGH'}
    
    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    #vsetko co treba pre zavretie 
    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        self.vid.release()
        cv2.destroyAllWindows() 


class MyProperties(bpy.types.PropertyGroup):
    """Property Group containing custom properties"""
    # misc bone names input
    head: bpy.props.StringProperty(name = "Head", default = "head", description="Bone that rotates head") 
    chin: bpy.props.StringProperty(name = "Chin", default = "chin", description="Bone controlling the lowest part of chin") 

    # mouth bone names input
    mouth_r: bpy.props.StringProperty(name = "Mouth - Right", default = "lips.R", description="Bone controlling the right mouth corner") 
    mouth_br: bpy.props.StringProperty(name = "Mouth - Bottom right", default = "lip.B.R.001", description="Bone controlling the right part of bottom lip") 
    mouth_b: bpy.props.StringProperty(name = "Mouth - Botton", default = "lip.B", description="Bone controlling the middle part of bottom lip") 
    mouth_bl: bpy.props.StringProperty(name = "Mouth - Bottom left", default = "lip.B.L.001", description="Bone controlling the left part of bottom lip") 
    mouth_l: bpy.props.StringProperty(name = "Mouth - Left", default = "lips.L", description="Bone controlling the left mouth corner")
    mouth_tl: bpy.props.StringProperty(name = "Mouth -Top Left", default = "lip.T.L.001",description="Bone controlling the left part of top lip") 
    mouth_t: bpy.props.StringProperty(name = "Mouth - Top", default = "lip.T", description="Bone controlling the middle part of top lip") 
    mouth_tr: bpy.props.StringProperty(name = "Mouth - Top Right", default = "lip.T.R.001", description="Bone controlling the right part of top lip") 

    # brow bone names input
    brow_l1: bpy.props.StringProperty(name = "Brow - Left (outer)", default = "brow.T.L.001", description="Bone controlling the outer part of the left eyebrow")
    brow_l2: bpy.props.StringProperty(name = "Brow - Left (middle)", default = "brow.T.L.002", description="Bone controlling the middle part of the left eyebrow")
    brow_l3: bpy.props.StringProperty(name = "Brow - Left (inner)", default = "brow.T.L.003", description="Bone controlling the inner part of the left eyebrow")
    brow_r3: bpy.props.StringProperty(name = "Brow - Right (inner)", default = "brow.T.R.003", description="Bone controlling the inner part of the right eyebrow")
    brow_r2: bpy.props.StringProperty(name = "Brow - Right (middle)", default = "brow.T.R.002", description="Bone controlling the middle part of the right eyebrow")
    brow_r1: bpy.props.StringProperty(name = "Brow - Right (outer)", default = "brow.T.R.001", description="Bone controlling the outer part of the right eyebrow")

    # jaw bone name input
    jaw: bpy.props.StringProperty(name = "Jaw", default = "jaw_master", description="Bone controlling the jaw")

    # nose bone name input
    nose_l: bpy.props.StringProperty(name = "Nose - Left", default = "nose.L.001", description="Bone controlling the left side of the nose")
    nose_m: bpy.props.StringProperty(name = "Nose - Tip", default = "nose.002", description="Bone controlling the tip of the nose")
    nose_r: bpy.props.StringProperty(name = "Nose - Right", default = "nose.R.001", description="Bone controlling the right side of the nose")

    # eye bone name input
    eye_l: bpy.props.StringProperty(name = "Eye - Left (outer corner)", default = "lid.B.L", description="Bone controlling the outer corner of left eye")
    eye_tl: bpy.props.StringProperty(name = "Eye - Top left", default = "lid.T.L.002", description="Bone controlling the upper lid of the left eye")
    eye_bl: bpy.props.StringProperty(name = "Eye - Bottom left", default = "lid.B.L.002", description="Bone controlling the lower lid of the left eye")

    eye_r: bpy.props.StringProperty(name = "Eye - Right (outer corner)", default = "lid.B.R", description="Bone controlling the outer corner of right eye")
    eye_tr: bpy.props.StringProperty(name = "Eye - Top right", default = "lid.T.R.002", description="Bone controlling the upper lid of the right eye")
    eye_br: bpy.props.StringProperty(name = "Eye - Bottom right", default = "lid.B.R.002", description="Bone controlling the lower lid of the right eye")

    # input for type of keyframe thats inserted
    keyframe_enum: bpy.props.EnumProperty(
        name = "Type", items= [('Location',"Location",""),
                                    ('Rotation',"Rotation",""),
                                    ('Scaling',"Scale",""),
                                    ('BUILTIN_KSI_LocRot', "Location & Rotation",""),
                                    ('LocRotScale',"Location, Rotation & Scale",""),
                                    ('WholeCharacter',"Whole Character",""),
        ],
        description="Choose the type of keyframe inserted"
    )

    # bools for rigify show options
    show_names: bpy.props.BoolProperty(name = "Show names", default = False, update=update_bool, description="Show names of bones")
    show_shapes: bpy.props.BoolProperty(name = "Show shapes", default = True, update=update_bool, description="Show shapes of the bones")
    show_col: bpy.props.BoolProperty(name = "Show group colors", default = True, update=update_bool, description="Show group colors of bones")
    show_front: bpy.props.BoolProperty(name = "Show in front", default = True,update=update_bool, description="Show bones in front of their assigned mesh. Only available when armature has been chosen.")
    show_axes: bpy.props.BoolProperty(name = "Show axes", default = False, update = update_bool, description="Show local axes of bones")
    # float pre axis offset
    axes_offset: bpy.props.FloatProperty(name = "Offset: ", default = 0, precision=2, min = 0, max = 1, update=update_bool, description="Change position of the bone axes. Increasing value moves them to the tip of the bone, decreasing moves them towards the head.")
    

class ArmtInput_PT_armt(bpy.types.Panel):
    """Panel containing armature options."""
    bl_label = "Armature"
    bl_idname = "ARMT_PT_armt_input"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AnimFace"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        col = layout.column()
        col.label(text ="Choose armature")
        col.prop(scene, "my_collection")
        
        col = layout.column()
        col.enabled = True if scene.my_collection else False
        col.prop(scene, "my_collection_objects")

        col = layout.column()
        col.label(text="Bone names")
        col.enabled = True if scene.my_collection_objects else False
        col.prop(mytool, "head")
        col.prop(mytool, "chin")

        col.prop(mytool, "mouth_r")
        col.prop(mytool, "mouth_br")
        col.prop(mytool, "mouth_b")
        col.prop(mytool, "mouth_bl")
        col.prop(mytool, "mouth_l")
        col.prop(mytool, "mouth_tl")
        col.prop(mytool, "mouth_t")
        col.prop(mytool, "mouth_tr")

        col.prop(mytool, "brow_l1")
        col.prop(mytool, "brow_l2")
        col.prop(mytool, "brow_l3")
        col.prop(mytool, "brow_r3")
        col.prop(mytool, "brow_r2")
        col.prop(mytool, "brow_r1")

        col.prop(mytool, "jaw")

        col.prop(mytool, "nose_l")
        col.prop(mytool, "nose_m")
        col.prop(mytool, "nose_r")

        col.prop(mytool, "eye_l")
        col.prop(mytool, "eye_tl")
        col.prop(mytool, "eye_bl")
        col.prop(mytool, "eye_r")
        col.prop(mytool, "eye_tr")
        col.prop(mytool, "eye_br")


class Control_PT_main(bpy.types.Panel):
    """Panel containing the controls."""
    bl_label = "Control"
    bl_idname = "MAINPANEL_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AnimFace"
    bl_options = {"DEFAULT_CLOSED"}
    

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        col1 = layout.column(align=False) 
        col1.label(text = "Calibrate")
        col1.operator("calibration.button", text = "Calibration")

        col2 = layout.column(align=False, heading="Capture video")
        col2.label(text="Capture")
        row = col2.row(align = False, heading="Capture video") 
        row.operator("start.button", text = "Start")
        row.operator("end.button", text = "End")

        col3= layout.column(align = False, heading="Insert keyframes")
        col3.prop(mytool, "keyframe_enum")
        label = "Stop recording" if isRecording else "Record movement"
        col3.operator("contkeyframe.button", text = label)
        col3.operator("singlekeyframe.button", text = "Insert keyframe")

class RigifyOpt_PT_rigify(bpy.types.Panel):
    """Panel containing Rigify options."""
    bl_label = "Rigify"
    bl_idname = "RGF_PT_rigify_opt"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AnimFace"
    bl_options = {"DEFAULT_CLOSED"}
    

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool

        col1 = layout.column(align = False,heading = "Viewport Display Options")
        col1.prop(mytool, "show_names")
        col1.prop(mytool, "show_shapes")
        col1.prop(mytool, "show_col")
        row = col1.row()
        row.enabled = True if scene.my_collection_objects else False
        row.prop(mytool, "show_front")
        row2= col1.row()
        row2.prop(mytool, "show_axes")
        row2.prop(mytool, "axes_offset")
 
        col2 = layout.column(align=False)
        col2.label(text ="Layer visibility")
        col2.operator("rigifyface.button", text = "Face", depress = faceVisible)
        row = col2.row()
        row.operator("rigifyprimface.button", text = "Face (Primary)", depress = primfaceVisible)
        row.operator("rigifysecface.button", text = "Face (Secondary)", depress = secfaceVisible)
        col2.operator("rigifytorso.button", text = "Torso", depress = headVisible)
        

class CalibrationButton(bpy.types.Operator):
    """Operator that starts calibration"""
    bl_idname = "calibration.button"
    bl_label = "Calibrate"
    bl_description = "Start calibration"

    def execute(self, context):
        calibrate(context)
        return {'FINISHED'} 


class StartButton(bpy.types.Operator):
    """Operator that starts video capture."""
    bl_idname = "start.button"
    bl_label = "Start"
    bl_description = "Start capture"


    def execute(self, context):
        global isRunning
        global armt
        global pose_bones
        global error_message

        isRunning = True 
        armt = bpy.context.scene.my_collection_objects # armature variable
        if armt is None: 
            self.report({"ERROR"}, "Armature has not been set. Please choose your armature.")
            return {"CANCELLED"}
        
        pose_bones.clear() 

        init_bone_positions() 
        if len(error_message) != 0:
            self.report({"ERROR"}, error_message)
            error_message = ''
            return {"CANCELLED"}
        
        # reset positions to base pose
        bpy.ops.pose.select_all(action = 'SELECT')
        bpy.ops.pose.user_transforms_clear() 
        bpy.ops.pose.select_all(action='DESELECT')

        bpy.ops.wm.modal_timer_operator() #
        return {'FINISHED'}
    
class EndButton(bpy.types.Operator):
    """Operator that ends video capture."""
    bl_idname = "end.button"
    bl_label = "End"
    bl_description = "End capture. Alternatively press ESC"
    

    def execute(self, context):
        global isRunning
        isRunning = False

        # reset positions to base pose
        bpy.ops.pose.select_all(action = 'SELECT')
        bpy.ops.pose.user_transforms_clear() 
        bpy.ops.pose.select_all(action='DESELECT')
        return {'FINISHED'}
    
class SingleKeyframeButton(bpy.types.Operator):
    """Operator that inputs a single keyframe of type chosen by user."""
    bl_idname = "singlekeyframe.button"
    bl_label = "Single Keyframe"
    bl_description = "Add a single keyframe of type chosen in Type"


    def execute(self, context):
        if not isRunning:
            self.report({"WARNING"}, "Please start capture to create animation.")
            return {"CANCELLED"}
        global kf_type
        bpy.ops.pose.select_all(action='SELECT')
        kf_type = bpy.context.scene.my_tool.keyframe_enum # keyframe type
        bpy.ops.anim.keyframe_insert_by_name(type=kf_type) 
        bpy.context.scene.frame_current += 1 
        bpy.ops.pose.select_all(action='DESELECT')
        
        return {'FINISHED'} 

class ContKeyframeButton(bpy.types.Operator):
    """Operator that starts and ends continuous recording of keyframes."""
    bl_idname = "contkeyframe.button"
    bl_label = "Continuous Keyframes"
    bl_description = "Record movement animation in real time"

    
    def execute(self, context):
        if not isRunning:
            self.report({"WARNING"}, "Please start capture to create animation.")
            return {"CANCELLED"}
        global isRecording
        global kf_type
        
        kf_type = bpy.context.scene.my_tool.keyframe_enum 
        if isRecording: # if animation is recording 
            isRecording = False
            if len(rec_animation) != 0:
                create_anim()
            return {'FINISHED'}
        
        isRecording = True
        return {'FINISHED'}
    

class RigifyFaceButton(bpy.types.Operator):
    """Operator that controls visibility of the Rigify Face layer."""
    bl_idname = "rigifyface.button"
    bl_label = "Face layer"
    bl_description = "Armature face layer visibility"

    def execute(self, context):
        global faceVisible
        if faceVisible:
            faceVisible = False
            bpy.data.armatures["Human.rigify"].layers[0] = False
            return {'FINISHED'}

        faceVisible = True
        bpy.data.armatures["Human.rigify"].layers[0] = True
        return {'FINISHED'}
    
class RigifyPrimFaceButton(bpy.types.Operator):
    """Operator that controls visibility of the Rigify Face (primary) layer."""
    bl_idname = "rigifyprimface.button"
    bl_label = "Primary face layer"
    bl_description = "Armature primary face layer visibilty"

    def execute(self, context):
        global primfaceVisible
        if primfaceVisible:
            primfaceVisible = False
            bpy.data.armatures["Human.rigify"].layers[1] = False
            return {'FINISHED'}

        primfaceVisible = True
        bpy.data.armatures["Human.rigify"].layers[1] = True
        return {'FINISHED'}

class RigifySecFaceButton(bpy.types.Operator):
    """Operator that controls visibility of the Rigify Face (secondary) layer."""
    bl_idname = "rigifysecface.button"
    bl_label = "Secondary face layer"
    bl_description = "Armature secondary face layer visibilty"

    def execute(self, context):
        global secfaceVisible
        if secfaceVisible:
            secfaceVisible = False
            bpy.data.armatures["Human.rigify"].layers[2] = False
            return {'FINISHED'}

        secfaceVisible = True
        bpy.data.armatures["Human.rigify"].layers[2] = True
        return {'FINISHED'}
    
class RigifyTorsoButton(bpy.types.Operator):
    """Operator that controls visibility of the Rigify Torso layer."""
    bl_idname = "rigifytorso.button"
    bl_label = "Torso layer"
    bl_description = "Armature torso layer visibilty"

    def execute(self, context):
        global headVisible
        if headVisible:
            headVisible = False
            bpy.data.armatures["Human.rigify"].layers[3] = False
            return {'FINISHED'}

        headVisible = True
        bpy.data.armatures["Human.rigify"].layers[3] = True
        return {'FINISHED'}
    

classes = (
    MyProperties,
    ArmtInput_PT_armt,
    Control_PT_main,
    RigifyOpt_PT_rigify,
    StartButton,
    ModalTimerOperator,
    EndButton,
    CalibrationButton,
    SingleKeyframeButton,
    ContKeyframeButton,
    RigifyFaceButton,
    RigifyPrimFaceButton,
    RigifySecFaceButton,
    RigifyTorsoButton,
)

# ----------------------------------------------------------------------------------------------------------------- #
def menu_func(self, context):
    self.layout.operator(ModalTimerOperator.bl_idname, text=ModalTimerOperator.bl_label)

def register():
    """Registers all classes to be ready for use."""
    for cls in preference_classes:
        bpy.utils.register_class(cls)

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=MyProperties)
    bpy.types.VIEW3D_MT_view.append(menu_func)
    bpy.types.Scene.my_collection = bpy.props.PointerProperty(
        name="Collection",
        type=bpy.types.Collection, description="Choose collection that contains desired armature")
    bpy.types.Scene.my_collection_objects = bpy.props.PointerProperty(
        name="Object",
        type=bpy.types.Object,
        poll=filter_callback, description="Choose armature to control")

def unregister():
    """Unregisters all classses."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.my_collection_objects
    for cls in reversed(preference_classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.my_collection
    bpy.types.VIEW3D_MT_view.remove(menu_func)
    del bpy.types.Scene.my_tool
    

if __name__ == "__main__":
    register()
