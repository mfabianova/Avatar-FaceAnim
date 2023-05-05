bl_info = {
    "name": "AnimateFace",
    "author": "Miriam Fabianova",
    "version": (1, 0),
    "blender": (3, 3, 1),
    "location": "",
    "description": "Animates an avatar face",
    "warning": "",
    "doc_url": "",
    "category": "Generic",
}

import bpy
import cv2, time
import mediapipe as mp
import copy
import math
import numpy as np
from pyquaternion import Quaternion
from mediapipe.framework.formats import landmark_pb2

#---------------------global variables----------------------------------------------------------
#booly ktore urcuju viditelnost vrstiev
faceVisible = True
primfaceVisible = True
secfaceVisible = True
headVisible = True

isRunning = False #zacanie nahravania videa (nepocita sa kalibracia)
isRecording = False #zacanie nahravania keyframeov

d1 = 0 #vzdialenost medzi ocami v prvom snimku
#rozmery obrazu kamery
height = 0
width = 0 
noise = 0 #sum v ramci crt tvare

#premenne urcene pouzivatelom v UI - kostra a typ klucovych snimkov
armt = None
kf_type = None 

#indexy kosti na detekovanej tvari
#pary k nim su [name]_bones inicializovane vo fcii init_bone_positions
mouth_indices = [76, 179, 15, 403, 292, 271, 12, 41] 
brow_indices = [276, 282, 285, 55, 52, 46] 
jaw_indices = [152]
nose_indices = [439, 0, 219]
eye_indices = [386, 253, 159, 23]
indices = [mouth_indices, brow_indices, jaw_indices, eye_indices, nose_indices]

pose_bones = [] #pozicie kosti v pose mode
head_bone = None #kost hlavy v pose mode

old_positions = [] #prvotne pozicie kosti v edit mode potom sa prepisuju na pozicie v predchadzajucom snimku
polygon_edit = [] #polygon v priestore avatara

#globalne premenne pre face detection
mpDraw = mp.solutions.mediapipe.solutions.drawing_utils
mpFaceMesh = mp.solutions.mediapipe.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode = False, max_num_faces=1, refine_landmarks = True) 

#zoznam nahranch snimkov
rec_animation = []

#nazov kosti, ktora ma nespravne meno - chybova hlaska
error_name = ''
error_message = ''


#------------------------------------------funkcie-------------------------------------------------------
def filter_callback(self, object):
    return object.name in self.my_collection.objects.keys()

def update_bool(self,context): 
    #updatovanie boolov v casti rigify - volana pri kazdej zmene
    bpy.data.armatures["Human.rigify"].show_names = bpy.context.scene.my_tool.show_names
    bpy.data.armatures["Human.rigify"].show_bone_custom_shapes = bpy.context.scene.my_tool.show_shapes
    bpy.data.armatures["Human.rigify"].show_group_colors = bpy.context.scene.my_tool.show_col
    bpy.data.armatures["Human.rigify"].show_axes = bpy.context.scene.my_tool.show_axes
    bpy.data.armatures["Human.rigify"].axes_position = bpy.context.scene.my_tool.axes_offset
    if bpy.context.scene.my_collection_objects is not None:
        bpy.data.objects[bpy.context.scene.my_collection_objects.name].show_in_front = bpy.context.scene.my_tool.show_front

def create_normalized_landmark_list(landmarks):
    normalized_landmarks = list(map(lambda p: landmark_pb2.NormalizedLandmark(x=p[0], y=p[1], z=p[2]), landmarks))
    return landmark_pb2.NormalizedLandmarkList(landmark=normalized_landmarks)

def triangle_normal(p1, p2, p3):
    #vrati normalu trojuholnika p1p2p3
    e1 = (p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2]) #hrana e1 = p3p1
    e2 = (p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2]) #hrana e2 = p3p2
    return np.cross(e2,e1) #vektorovy sucin hran

def dist(p1, p2):
    #vrati euklidovsku vzdialenost dvoch bodov
    if len(p1) == 3:
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    elif len(p2) == 2:
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    else:
        return
    
def normalize(v): 
    #vrati normalizovany vektor vektora v
    s = 0 #sucet druhych mocnin
    for i in range(len(v)):
        s += v[i]**2
    v_norm = v/np.sqrt(s)
    return v_norm

def calibrate(context):
    #premenne, ktore sa urcuju pocas kalibracie - vzdialenost medzi ocami v prvom snimku, velkost obrazu, noise
    global d1
    global height, width
    global noise

    first_time = True #prvy snimok
    vid = cv2.VideoCapture(0) #spustenie nahravania
    while(True):
        #precitanie snimku
        succ, frame = vid.read() 
        height, width = frame.shape[:2] 
        flipped = cv2.flip(frame, 1)

        #zobrazovanie
        cv2.putText(flipped, "Look straight into the camera", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,127,0), thickness=2, lineType=cv2.LINE_AA) 
        cv2.putText(flipped, "and press c to calibrate",      (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,127,0), thickness=2, lineType=cv2.LINE_AA) 
        cv2.rectangle(flipped, (int(width/2 - 70), int(height/2 - 100)), (int(width/2 + 70), int(height/2 + 100)), (0,255,0), 2)
        cv2.line(flipped, (int(width/2 - 10), int(height/2 - 10)), (int(width/2 + 10), int(height/2 + 10)), (0,255,0), 2)
        cv2.line(flipped, (int(width/2 - 10), int(height/2 + 10)), (int(width/2 + 10), int(height/2 - 10)), (0,255,0), 2)
        cv2.imshow("Calibration", flipped)

        #detekcia crt v snimku, ak neuspesne - preskocit vsetky prikazy dalej
        full_lm = get_face_lm(flipped)
        if full_lm is None: 
            continue
        
        #ak stlaci pouzivatel c, kamera sa vypne a ulozia sa aktualne hodnoty d1, noise
        if(cv2.waitKey(1) & 0xFF == ord('c')):
            eye_centerR = ((full_lm.landmark[159].x + full_lm.landmark[145].x)*width/2, (full_lm.landmark[159].y + full_lm.landmark[145].y)*height/2, (full_lm.landmark[159].z + full_lm.landmark[145].z)*width/2) #priemer landmarkov 159 a 145
            eye_centerL = ((full_lm.landmark[386].x + full_lm.landmark[374].x)*width/2, (full_lm.landmark[386].y + full_lm.landmark[374].y)*height/2, (full_lm.landmark[386].z + full_lm.landmark[374].z)*width/2) #priemer landmarkov 386 a 374 
            d1 = dist(eye_centerL, eye_centerR) #inicialna vzdialenost zreniciek
            break
            
    vid.release()
    cv2.destroyAllWindows()

def get_angle(p1, p2, p3):
    #vrati uhol p1p2p3
    v0 = (p2[0] - p1[0], p2[1]-p1[1]) #vektor p1p2
    v1 = (p2[0] - p3[0], p2[1] - p3[1]) #vektor p3p2

    #normy vektorov
    v0_norm = np.sqrt(v0[0]**2 + v0[1]**2) 
    v1_norm = np.sqrt(v1[0]**2 + v1[1]**2)

    #uhol
    angle = np.arccos((v0[0] * v1[0] + v0[1]*v1[1])/(v0_norm*v1_norm))
    return angle


def mean_value_bc(point, vertices): 
    n = len(vertices)
    EPS = 1e-5
    
    s = n * [(0.0, 0.0)] # vektory od vrcholov k bodu
    r = n * [0.0]        # vzdialenosti od vrcholov k bodu
    area2 = n * [0.0]    # plocha * 2
    dot_prod = n * [0.0] # skalarny sucin
    w = n * [0.0]        # vysledne barycentricke suradnice
    
    for i in range(n):
        s[i] = (vertices[i][0] - point[0], vertices[i][1] - point[1])
        r[i] = np.sqrt(s[i][0]**2 + s[i][1]**2) 
        if abs(r[i]) < EPS: # je bod blizko nejakeho vrcholu ?
            w[i] = 1.0
            return w
    
    for i in range(n):
        ip1 = (i + 1) % n # nasledujuci index vrcholu
        area2[i] = (s[i][0]*s[ip1][1] - s[ip1][0]*s[i][1])/2
        dot_prod[i] = s[i][0]*s[ip1][0] + s[i][1]*s[ip1][1]
    
    sum_w = 0
    
    for i in range(n):
        ip1 = (i + 1) % n            # nasledujuci index vrcholu
        im1 = ((i - 1) % n + n) % n  # predchadzajuci index vrcholu
        if abs(area2[ip1]) > EPS:   # je bod dostatocne daleko od hrany ?
            w[i] += (r[im1] - dot_prod[im1] / r[i]) / area2[im1] # vypocitaj zaklad baryc. suradnice
        if abs(area2[im1]) > EPS:   # je bod dostatocne daleko od hrany ?
            w[i] += (r[ip1] - dot_prod[i] / r[i]) / area2[i]     # vypocitaj zaklad baryc. suradnice
        else: #ak je bod pri hrane
            w = np.zeros(len(vertices))
            w[i] = r[ip1]/(r[ip1] + r[i])
            w[ip1] = r[i]/(r[ip1] + r[i]) #tu uz je sum 1
            return w
        sum_w += w[i]
            
    for i in range(n):
        w[i] /= sum_w # normalizacia na standardne baryc. suradnice

    return w

def get_face_lm(_frame):  
    rgbFrame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB) #snimok je bgr - konverzia 
    
    #-----------------detekcia--------------------
    _frame.flags.writeable = False #improve perf
    landmarks = faceMesh.process(rgbFrame) 
    _frame.flags.writeable = True
    
    full_lm = landmarks.multi_face_landmarks[0] 
    return full_lm 
                    

def get_mesh_normal(_landmarks):
    #funkcia vrati normalu meshu ako priemer normal niekolkych jej trojuholnikov
    #indexy crt, ktore pouzivame na pocitanie normaly
    indices = [332, 389, 323, 397, 378, 149, 172, 93, 162, 103] #pred 149 bola 152 - omitted, helps with expressions not changing rotation of head
                                                                #na zaciatku 10
    #indices = [93, 234, 197, 454, 323]

    neighbour_lm = [] #crty tvare s indexami v indices
    for i in indices:
        neighbour_lm.append((_landmarks[i][0] * width, _landmarks[i][1] * height, _landmarks[i][2] * width))
    
    triangle_normals = [] #normaly trojuholnikov
    #pocitanie normal pre kazdy trojuholnik z neighbour_lm
    for i in range(len(neighbour_lm)):
        curr_n = triangle_normal(neighbour_lm[(i-1)%len(indices)],neighbour_lm[i], (_landmarks[1][0] * width, _landmarks[1][1] * height, _landmarks[1][2] * width))
        triangle_normals.append(curr_n)

    #priemer normal trojuholnikov
    return np.mean(np.array(triangle_normals), axis=0)

def rotate_xy(_lm, _normal, _rotation_pt):
    #funckia rotuje landmarky tak, aby normala smerovala vpred
    #vystupom je zoznam landmarkov, uhol a vektor pomocou ktorych rotujeme

    rotated_lm = [_lm[0]] #prvy nos do zoznamu
    normalized = normalize(_normal) #normalizacia normaly meshu
    
    rotation_vector = np.cross(normalized, (0,0,1)) #vektor okolo ktoreho rotujeme
    angle = np.arccos(np.dot((0,0,1), normalized)) #uhol rotacie

    #inicializacia kvaterniona
    q =  Quaternion(axis=[rotation_vector[0], rotation_vector[1], rotation_vector[2]], angle = angle)
    for i in range(468):
        if i != 1: 
            #rotacia 
            lm_vec = (_lm[i][0] - _rotation_pt[0],_lm[i][1] - _rotation_pt[1],_lm[i][2] - _rotation_pt[2]) #vektor s pociatkom v nose a koncom v danom landmarku = vektor, ktory rotujeme
            rotated_vec = q.rotate(lm_vec) #vysledny vektor prvej rotacie
            rotated_lm.append((_rotation_pt[0] + rotated_vec[0], _rotation_pt[1] + rotated_vec[1], _rotation_pt[2] + rotated_vec[2])) #ulozenie rotovaneho landmarku do zoznamu 
         
    return rotated_lm, angle, rotation_vector

def translate(_lm, _vec):
    #funkcia posunie crty tvare tak aby landmark nosa bol (0,0)
    translated_lm = [(0.0, 0.0)] #nos je prvy
    for i in range(1,468):
        landmark = (_lm[i][0] + _vec[0], _lm[i][1] + _vec[1])
        translated_lm.append(landmark)
    return translated_lm

def scale(_lm, _fac): 
    #funckia skaluje landmarky _lm skalovacim faktorom _fac
    translated_nose = _lm[0]
    scaled_lm = [translated_nose]
    for i in range(1, 468):
        landmark = (_fac * (_lm[i][0] - translated_nose[0]) + translated_nose[0], _fac * (_lm[i][1] - translated_nose[1]) + translated_nose[1]) 
        scaled_lm.append(landmark)
        
    return scaled_lm

def rotate_z(_lm, _forehead):
    #funkcia rotuje landmarky tak, aby bol vektor nos - celo (0,-1)

    rotated_z_lm = [(0,0)]
    forehead_angle = np.arctan((_forehead[0])/np.abs(_forehead[1])) #hladanie uhla rotacie

    if _forehead[1] > 0: #ak vektor nos - celo smeruje dole - upravujeme uhol
        forehead_angle = math.pi - forehead_angle

    for i in range(1, 468):
        vec = (_lm[i][0], _lm[i][1]) #vektor, ktory rotujeme - rovnako ako prva rotacia
        x =  math.cos(forehead_angle) * vec[0] + math.sin(forehead_angle) * vec[1]
        y = -math.sin(forehead_angle) * vec[0] + math.cos(forehead_angle) * vec[1] #samotna rotacia
                        
        landmark = (x, y)
        rotated_z_lm.append(landmark)
    return rotated_z_lm


def bone_movement(_lm, _indices, _polygon_img, _polygon_edit, _old_positions, _pose_bones, group = None):
    #funkcia najde nove pozicie na ktorych maju byt kosti, podla toho pohne
    #vrati nove pozicie, aby sa prepisali na stare
    
    future_old_positions = [] #zoznam do ktoreho dame nove pozicie v edit mode aby sme ich mohli po cykle prepisat ako stare
    group_diff = []
    for i in range(len(_indices)): #prechadzame vsetkymi kostami/landmarkami ust
        coor = mean_value_bc(_lm[_indices[i]], _polygon_img) #vypocitame suradnice

        #vypocitame miesto v priestore avatara kde ma byt nova pozicia bodu
        new_position = (0,0)
        for j in range(len(_polygon_edit)):
            new_position += _polygon_edit[j] * coor[j]  
        future_old_positions.append(new_position) #ulozime novu poziciu do zoznamu 

        #stara pozicia sa nacita zo zoznamu starych pozicii
        old_position = _old_positions[i]
        difference = (new_position[0]-old_position[0], new_position[1]-old_position[1]) #vypocita sa rozdiel stara - nova pozicia

        #diferencia sa upravuje ak sa jedna o kosti ust alebo celuste
        if group == 'MOUTH':
            if i in {1,3,5,7}: #vedlajsie kosti (B.L, B.R, T.L a T.R sa hybu uz s pohybmi ostatnych)
                difference = (0.5 * difference[0], 0.5 * difference[1])
    
            if 0<i and i<4: #spodne kosti hybe aj celust
                difference = (0.4 * difference[0], 0.4 * difference[1])
        elif group == 'JAW': #kost celuste ma opacne orientovanu yovu os
            difference = (0.0, -difference[1])
        elif group == 'EYE':
            difference = (1.25 * difference[0], 1.25 * difference[1])
        group_diff.append(difference)
        #samotne hybanie kosti
        _pose_bones[i].location = (_pose_bones[i].location.x + difference[0],_pose_bones[i].location.y+ difference[1], _pose_bones[i].location.z)
            #future_old_positions.append(new_position) #ulozime novu poziciu do zoznamu 
        #else:
            #future_old_positions.append(new_position) #ulozime novu poziciu do zoznamu 
                        
    #vrat stare kosti
    return future_old_positions, group_diff

def one_euro(_lm, _isFirst, _dx_prev, _x_prev):
    #funkcia vyhladzuje pohyb hlavy
    span = 10
    f_cutoff = 0.01
    threshold =  1.25
    dx_scale = 600
    #parametre rozne pre pohyb hlavy a pre pohyb tvare
    x = np.array(_lm)
    if _isFirst:
        _dx_prev = np.array(len(_lm) * [(0.0,0.0,0.0)]) #vytvori list nul s dlzkou landmarkov
        _x_prev = x #predchadzajuci list
    
    dx = dx_scale * (x -_x_prev) 
    
    alpha_dc = 2 / (1 + span)
    dx_hat = np.add(alpha_dc * dx, (1 - alpha_dc) * _dx_prev) #interpolacia medzi rozdielmi

    cutoff = f_cutoff + np.abs(dx_hat)
    alpha_fc = 2 / (1 + span / cutoff)
    x_hat = np.around(np.where(np.abs(dx) < threshold, _x_prev, alpha_fc * x + (1 - alpha_fc) * _x_prev), decimals=3)
    
    return x_hat, dx_hat

def one_euro_head(_lm, _isFirst, _dx_prev, _x_prev, _threshold, _dx_scale):
    #funkcia vyhladzuje pohyb tvare

    span = 7
    f_cutoff = 0.01
    #parametre rozne pre pohyb hlavy a pre pohyb tvare, dx_scale a _threshold    
    x = np.array(_lm) 
    if _isFirst:
        _dx_prev = np.array(len(_lm) * [0.0]) #vytvori list nul s dlzkou landmarkov
        _x_prev = x #predchadzajuci list

    dx = _dx_scale * np.add(x, -_x_prev) 
    
    alpha_dc = 2 / (1 + span)
    dx_hat = np.add(alpha_dc * dx, (1 - alpha_dc) * _dx_prev) #interpolacia medzi rozdielmi

    cutoff = f_cutoff + np.abs(dx_hat)
    alpha_fc = 2 / (1 + span / cutoff)
    x_hat = np.add(alpha_fc * x, (1 - alpha_fc) * _x_prev) #zas interpolacia medzi listami

    x_hat = np.where(np.abs(dx) < _threshold, _x_prev, np.add(alpha_fc * x, (1 - alpha_fc) * _x_prev))

    return x_hat, dx_hat

def init_bone_positions():
    #inicializuje kostru, polygon v priestore avatara a pozicie kosti na zaciatku 
    global mouth_bones, brow_bones, jaw_bones, nose_bones, eye_bones #mena kosti
    global error_message

    global pose_bones #pozicie kosti v pose mode
    global head_bone #kost hlavy v psoe mode
    global old_positions #pozicie kosti na zaciatku
    global polygon_edit #polygon v priestore avatara
    
    bpy.ops.object.mode_set(mode='EDIT')
    #kostra v edit mode
    ob = bpy.data.objects[armt.name]
    armature_edit = ob.data
    
    #vypisanie chybovej hlasky ak vybrany objekt nie je kostra
    try:
        editbones = armature_edit.edit_bones
    except Exception:
        error_message = 'The chosen object is not an armature.'
        return
   
    #vypisanie chybovej hlasky ak chybne menokosti
    tool = bpy.context.scene.my_tool
    string_prop = [tool.head, tool.chin, tool.mouth_r, tool.mouth_br, tool.mouth_b, tool.mouth_bl, tool.mouth_l, tool.mouth_tl, tool.mouth_t,
                   tool.brow_l1, tool.brow_l2, tool.brow_l3, tool.brow_r3, tool.brow_r2, tool.brow_r1, tool.jaw,
                   tool.nose_l, tool.nose_m, tool.eye_l, tool.eye_tl, tool.eye_bl, tool.eye_r,tool.eye_tr, tool.eye_br]
    for name in string_prop:
        bone = armature_edit.edit_bones.get(name)
        if bone is None:
            error_message = 'There is no bone with name ' + name +'.'
            return
        
    #nazvy kosti
    mouth_bones = [tool.mouth_r,tool.mouth_br,tool.mouth_b,tool.mouth_bl,tool.mouth_l,tool.mouth_tl, tool.mouth_t, tool.mouth_tr]
    brow_bones = [tool.brow_l1, tool.brow_l2,tool.brow_l3, tool.brow_r3, tool.brow_r2, tool.brow_r1] #zlava doprava z avatarovho pohladu 
    jaw_bones = [tool.jaw]
    nose_bones = [tool.nose_l, tool.nose_m, tool.nose_r] #lavy stred pravy z pohladu avatara
    eye_bones = [tool.eye_tl, tool.eye_bl, tool.eye_tr, tool.eye_br]#4 cervene kosti zhora dole, zlava doprava z pohladu avatara

    #kosti polygonu
    chinBone_edit = armature_edit.edit_bones[tool.chin]
    mouthBoneL_edit = armature_edit.edit_bones[tool.mouth_l]    
    noseBoneL_edit = armature_edit.edit_bones[tool.nose_l]   
    eyeBoneL_edit = armature_edit.edit_bones[tool.eye_l] 
    browBoneL3_edit = armature_edit.edit_bones[tool.brow_l3]
    browBoneR3_edit = armature_edit.edit_bones[tool.brow_r3]
    eyeBoneR_edit = armature_edit.edit_bones[tool.eye_r] 
    noseBoneR_edit = armature_edit.edit_bones[tool.nose_r]  
    mouthBoneR_edit = armature_edit.edit_bones[tool.mouth_r] 
    
    #zoznam kosti polygonu
    polygon_edit = np.array([(chinBone_edit.head.x, chinBone_edit.head.z),
                             (mouthBoneL_edit.head.x, mouthBoneL_edit.head.z),
                             (noseBoneL_edit.head.x, noseBoneL_edit.head.z),
                             (eyeBoneL_edit.head.x, eyeBoneL_edit.head.z),
                             (browBoneL3_edit.head.x, browBoneL3_edit.head.z),
                             (browBoneR3_edit.head.x, browBoneR3_edit.head.z),
                             (eyeBoneR_edit.head.x, eyeBoneR_edit.head.z),
                             (noseBoneR_edit.head.x, noseBoneR_edit.head.z),
                             (mouthBoneR_edit.head.x, mouthBoneR_edit.head.z)])
        
    #----------------kosti ust v edit mode---------------------
    mouth_old_positions = []
    for i in range(len(mouth_bones)):
        lmBone_edit = armature_edit.edit_bones[mouth_bones[i]]
        mouth_old_positions.append((lmBone_edit.head.x, lmBone_edit.head.z)) #pozicie kosti, obycajny append(kost.head) menilo pozicie po vnoreni do cyklu

    #----------------kost celuste v edit mode------------------
    jaw_old_positions = []
    for i in range(len(jaw_bones)):
        lmBone_edit = armature_edit.edit_bones[jaw_bones[i]]
        jaw_old_positions.append((lmBone_edit.tail.x, lmBone_edit.tail.z))

        #--------------kosti obocia v edit mode------------------
    brow_old_positions = []
    for i in range(len(brow_bones)):
        lmBone_edit = armature_edit.edit_bones[brow_bones[i]]
        brow_old_positions.append((lmBone_edit.head.x, lmBone_edit.head.z))

    #--------------kosti nosa v edit mode------------------
    nose_old_positions = []
    for i in range(len(nose_bones)):
        lmBone_edit = armature_edit.edit_bones[nose_bones[i]]
        nose_old_positions.append((lmBone_edit.head.x, lmBone_edit.head.z))
        
    #--------------kosti oci v edit mode------------------
    eye_old_positions = []
    for i in range(len(eye_bones)):
        lmBone_edit = armature_edit.edit_bones[eye_bones[i]]
        eye_old_positions.append((lmBone_edit.head.x, lmBone_edit.head.z))

    old_positions = [mouth_old_positions, brow_old_positions, jaw_old_positions, eye_old_positions, nose_old_positions]
        

#------------polohy kosti v pose mode---------------------------
    bpy.ops.object.mode_set(mode='POSE')

    armature_pose = bpy.context.scene.objects[armt.name] 

    #------------------kosti ust v pose mode---------------
    mouth_pose_bones = []
    for i in range(len(mouth_bones)):
        lmBone_pose = armature_pose.pose.bones.get(mouth_bones[i])
        mouth_pose_bones.append(lmBone_pose)

    #-------------------kost celuste v pose mode-----------------
    jaw_pose_bones  =[]
    for i in range(len(jaw_bones)):
        lmBone_pose = armature_pose.pose.bones.get(jaw_bones[i])
        jaw_pose_bones.append(lmBone_pose)    

    #------------------kosti obocia v pose mode---------------
    brow_pose_bones = []
    for i in range(len(brow_bones)):
        lmBone_pose = armature_pose.pose.bones.get(brow_bones[i])
        brow_pose_bones.append(lmBone_pose)

    #------------------kosti nosa v pose mode---------------
    nose_pose_bones = []
    for i in range(len(nose_bones)):
        lmBone_pose = armature_pose.pose.bones.get(nose_bones[i])
        nose_pose_bones.append(lmBone_pose)

    #------------------kosti oci v pose mode---------------
    eye_pose_bones = []
    for i in range(len(eye_bones)):
        lmBone_pose = armature_pose.pose.bones.get(eye_bones[i])
        eye_pose_bones.append(lmBone_pose)

    pose_bones = [mouth_pose_bones, brow_pose_bones, jaw_pose_bones, eye_pose_bones, nose_pose_bones]
                
    head_bone = armature_pose.pose.bones.get(bpy.context.scene.my_tool.head)

def create_anim():
    #vlozi klucove snimky podla nahranych differencii
    #transformacie do povodneho stavu
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.user_transforms_clear() 
    bpy.ops.pose.select_all(action='DESELECT')

    for all_diff in rec_animation: #all_diff je zoznam rozdielov pre cely frame
        for i in range(len(all_diff)): #prejdeme kazdou skupinou kosti (mouth, eye...) group_diff v all_diff
            group_diff = all_diff[i]
            for j in range(len(group_diff)): #kazdu kost v skupine posunieme o jej rozdiel
                if i != len(all_diff)-1: #ak nie sme v poslednej skupine (skupina ktora hybe hlavou)
                    pose_bones[i][j].location = (pose_bones[i][j].location.x + group_diff[j][0], pose_bones[i][j].location.y + group_diff[j][1], pose_bones[i][j].location.z)
                else:
                    head_bone.bone.select = True

                    #rotacia okolo osi x a y - kvaterniony
                    head_bone.rotation_mode = 'QUATERNION'
                    head_bone.rotation_quaternion = (group_diff[1], group_diff[2], -group_diff[3], group_diff[4])
                
                    #rotacia okolo osi z
                    #ak su kosti skyte, odkryju sa rotuju a skryju spat
                    isHidden = bpy.context.active_object.hide_get()
                    bpy.context.active_object.hide_set(False)
                    bpy.ops.transform.rotate(value = group_diff[0], orient_axis='Z', orient_type= 'LOCAL') #rotacia
                    if isHidden:
                        bpy.context.active_object.hide_set(True)
            
            
        #vlozenie klucovej snimky a posunutie o tri snimky dalej
        bpy.ops.pose.select_all(action='SELECT')
        isHidden = bpy.context.active_object.hide_get() 
        bpy.context.active_object.hide_set(False)
        bpy.ops.anim.keyframe_insert_by_name(type=kf_type) 
        bpy.context.scene.frame_current += 3
        if isHidden:
            bpy.context.active_object.hide_set(True)
            
        bpy.ops.pose.select_all(action='DESELECT')
        
    rec_animation.clear()#na konci vymazanie dat
    bpy.context.scene.frame_current = 0

        
#--------------------------------------------------------------------------------------

class ModalTimerOperator(bpy.types.Operator):
    """Operator which runs itself from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"
    
    
    global isRunning #spustenie nahravania videa (s pohybom, pri kalibracii je False)
    global isRecording #spustenie nahravania keyframeov
    global kf_type #typ ulozenia keyframeu, ktory si vyberie pouzivatel v UI

    global height, width #velkost videa
    global d1 #vzdialenost medzi ocami
    global noise #sum v ramci crt tvare

    global pose_bones #kosti v pose mode
    global head_bone #kosti hlavy v pose mode
    global old_positions #initially pozicie kosti v edit mode potom sa prepisuju na pozicie v predchadzajucom snimku

    global rec_animation

    #video capture premenna
    vid = None

    x_hat_prev_face = np.array(468 * [(0.0, 0.0,0.0)]) #listy pouzivane vo vyhladzovani tvare
    dx_hat_prev_face = np.array(468 * [(0.0,0.0,0.0)])
    x_hat_prev_face2 = np.array(468 * [(0.0, 0.0)]) #listy pouzivane vo vyhladzovani tvare
    dx_hat_prev_face2 = np.array(468 * [(0.0,0.0)])
    x_hat_prev_head = np.array(4 * [0.0]) #listy pouzivane vo vyhladzovani hlavy - rotacia x,y
    dx_hat_prev_head = np.array(4 * [0.0])
    x_hat_prev_z = np.array([0.0]) #listy pouzivane vo vyhladzovani hlavy - rotacia z
    dx_hat_prev_z = np.array([0.0])

    isFirst= True #boolean pre prvy snimok
    d1 = 0 #vzdialenost oci v prvom snimku

    counter = 0 #pocitanie snimkov pre zber dat 
    plot_points = [] #list pre nevyhladene data
    smooth_plot_points = [] #list pre vyhladene data
    
    polygon_indices = [152, 292, 439, 263, 285, 55, 33, 219, 76] #indexy pre landmarky prisluchajuce ku kostiam polygonu
    polygon_img = [] #pozicie landmarkov s indexami v polygon_indices
    
    prevTime = 0
    def modal(self, context, event):
        #ak stlacim tlacidla vypne sa
        if event.type in {'RIGHTMOUSE', 'ESC'} or isRunning == False:
            self.cancel(context)
            return {'CANCELLED'}

        #ak timer pokracuje nacita sa video najdu sa landmarky zobrazia sa a pohne sa kostou
        if event.type == 'TIMER':
            #ak pouzivatel nekalibroval, najprv spusti kalibraciu
            if width == 0:
                calibrate(context)

            #precitanie snimku vo videu
            if self.vid == None:
                self.vid = cv2.VideoCapture(0)
            succ, frame = self.vid.read() 
            flipped = cv2.flip(frame,1)

            #detekcia v snimku
            full_lm = get_face_lm(flipped) 
            
            #vyhladenie landmarkov
            smooth_lm, dx_list = one_euro(np.array([(lm.x, lm.y, lm.z) for lm in full_lm.landmark]), self.isFirst,self.dx_hat_prev_face, self.x_hat_prev_face)

            #ak boli detekovane ulozit do zoznamu predchadzajuce lm a vykreslenie
            if len(smooth_lm) != 0:
                self.x_hat_prev_face = copy.deepcopy(smooth_lm)
                self.dx_hat_prev_face = copy.deepcopy(dx_list)

                mpDraw.draw_landmarks(flipped, create_normalized_landmark_list(smooth_lm), mpFaceMesh.FACEMESH_CONTOURS,
                                 mpDraw.DrawingSpec(thickness = 1, circle_radius= 1, color = (0,255,0)), mpDraw.DrawingSpec(thickness = 1, color = (0,0,255)))

            #vypocet fps
            currTime = time.time()
            fps = int(1/(currTime - self.prevTime))
            self.prevTime = currTime

            #zobrazenie snimku
            cv2.putText(flipped, "fps = " + str(fps), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), thickness = 3)
            cv2.imshow('CameraVideo', flipped) 
            if full_lm is None:
                return {'PASS_THROUGH'}
            
            
        
#---------------------------urcovanie vzdialenosti od kamery-------------------------------------------------
            #vzdialenost od kamery = vzdialenost zreniciek v danom snimku / vzdialenost zreniciek pocas kalibracie
            #predpoklad = vzdialenost od kamery pocas kalibracie== 1
            eye_centerR = ((smooth_lm[159][0] + smooth_lm[145][0])*width/2, (smooth_lm[159][1] + smooth_lm[145][1])*height/2, (smooth_lm[159][2] + smooth_lm[145][2])*width/2) #priemer landmarkov 159 a 145
            eye_centerL = ((smooth_lm[386][0] + smooth_lm[374][0])*width/2, (smooth_lm[386][1] + smooth_lm[374][1])*height/2, (smooth_lm[386][2] + smooth_lm[374][2])*width/2) #priemer landmarkov 386 a 374

            #vypocet pomeru d = vzdialenost od kamery
            d = d1/dist(eye_centerL, eye_centerR)

#---------------------------urcovanie normaly na spicke nosa-------------------------------------------------- 
            nose_lm = (smooth_lm[1][0] * width, smooth_lm[1][1] * height, smooth_lm[1][2] * width) #landmark spicky nosa v suradniciach v obraze
            avg_normal = get_mesh_normal(smooth_lm)
#------------------------------------transformacie ostatnych landmarkov------------------------------------------------------
            
            #poradie transformacii: prva rotacia pomocou vytvoreneho kvaternionu
                                   #posunutie
                                   #skalovanie
                                   #rotacia okolo osi z (uz len v 2d)
            #transformacie nosa
            nose_proj = (nose_lm[0], nose_lm[1]) 
            translation_vec = (- nose_proj[0], - nose_proj[1]) #urcenie vektora posunutia (tak aby bol landmark nosa v (0,0)

            #-------------------obrazove sur landmarkov-------------------
            img_landmarks = []
            img_landmarks.append(nose_lm)
            for i in range(468):
                if i != 1: #landmark 0 nie je nos, landmark 1 je nos, preto treba preskocit i == 1 ked sa ukladaju landmarky prvy krat
                     #ukladanie landmarkov do zoznamu img_landmarks
                    landmark = (smooth_lm[i][0] * width, smooth_lm[i][1] * height, smooth_lm[i][2] * width)
                    img_landmarks.append(landmark) 
            #-----------------rotacia okolo xy-----------------
            rotated_landmarks, theta, rotation_vector = rotate_xy(img_landmarks, avg_normal, nose_lm)
            
            #-----------------translacia-----------------------
            translated_landmarks = translate(rotated_landmarks, translation_vec)

            #-----------------skalovanie-----------------------
            scaling_fac = d  
            scaled_landmarks = scale(translated_landmarks, scaling_fac)
            forehead = scaled_landmarks[10]
                    
            #---------------rotacia okolo z----------------------
            rotated_z_landmarks = rotate_z(scaled_landmarks, forehead)
            final_landmarks = rotated_z_landmarks
            #final_landmarks, dx = one_euro(rotated_z_landmarks, self.isFirst, self.dx_hat_prev_face2, self.x_hat_prev_face2,13, 0.01, 0.5, 10)
            #self.x_hat_prev_face2 = copy.deepcopy(final_landmarks)
            #self.dx_hat_prev_face2 = copy.deepcopy(dx)


#--------------------------------------------------------rotacia hlavy-------------------------------------------------------------------------
            #--------------------------------------------------------rotacia hlavy-------------------------------------------------------------------------
            #uhol oci - (1,0) podla tohto uhla rotujeme okolo osi z 
            #(oci v tomto pripade z prveho netransformovaneho listu landmarkov = img_landmarks)
            vec = (img_landmarks[263][0] - img_landmarks[33][0], img_landmarks[263][1]-img_landmarks[33][1]) #vektor oci
            alpha =  np.arctan((vec[1])/np.abs(vec[0]-1)) #uhol
            
            #vyber kosti hlavy
            head_bone.bone.select = True

            #rotacia okolo osi x a y - kvaterniony
            head_bone.rotation_mode = 'QUATERNION'
            head_bone.rotation_quaternion = (theta, rotation_vector[0], -rotation_vector[1], rotation_vector[2])
                
            #rotacia okolo osi z
            #ak su kosti skyte, odkryju sa rotuju a skryju spat
            isHidden = bpy.context.active_object.hide_get()
            bpy.context.active_object.hide_set(False)
            bpy.ops.transform.rotate(value = alpha, orient_axis='Z', orient_type= 'LOCAL') #rotacia
            if isHidden:
                bpy.context.active_object.hide_set(True)
            
            #ak prvy snimok vytvori sa polygon v obrazku
            if self.isFirst:     
                self.isFirst = False
                self.polygon_img.clear()
                for i in self.polygon_indices:
                    self.polygon_img.append(final_landmarks[i])
                
#--------------------------------------hybanie kostami-------------------------------------------------------------------------------
            
            opt = ['MOUTH', None, 'JAW', 'EYE', None] #usta a celust sa pohybuju specialne
            all_diff = []
            for i in range(len(indices)):
                old, diff = bone_movement(final_landmarks, indices[i], self.polygon_img, polygon_edit, old_positions[i], pose_bones[i], opt[i])
                old_positions[i] = copy.deepcopy(old)
                #diff su rozdiely v ramci jednej skupiny, tie ulozime do velkych differences ktore su v ramci celeho frame
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
            if isRecording: #kazdy frame sa nahrava, dat pouzivatelovi urcit ako casto?
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
    #misc bone names input
    head: bpy.props.StringProperty(name = "Head", default = "head", description="Bone that rotates head") #right
    chin: bpy.props.StringProperty(name = "Chin", default = "chin", description="Bone controlling the lowest part of chin") #right

    #mouth bone names input
    mouth_r: bpy.props.StringProperty(name = "Mouth - Right", default = "lips.R", description="Bone controlling the right mouth corner") #right
    mouth_br: bpy.props.StringProperty(name = "Mouth - Bottom right", default = "lip.B.R.001", description="Bone controlling the right part of bottom lip") #bottom right
    mouth_b: bpy.props.StringProperty(name = "Mouth - Botton", default = "lip.B", description="Bone controlling the middle part of bottom lip") #bottom
    mouth_bl: bpy.props.StringProperty(name = "Mouth - Bottom left", default = "lip.B.L.001", description="Bone controlling the left part of bottom lip") #bottom left
    mouth_l: bpy.props.StringProperty(name = "Mouth - Left", default = "lips.L", description="Bone controlling the left mouth corner") #left
    mouth_tl: bpy.props.StringProperty(name = "Mouth -Top Left", default = "lip.T.L.001",description="Bone controlling the left part of top lip") #top left
    mouth_t: bpy.props.StringProperty(name = "Mouth - Top", default = "lip.T", description="Bone controlling the middle part of top lip") #top
    mouth_tr: bpy.props.StringProperty(name = "Mouth - Top Right", default = "lip.T.R.001", description="Bone controlling the right part of top lip") #top right

    #brow bone names input
    brow_l1: bpy.props.StringProperty(name = "Brow - Left (outer)", default = "brow.T.L.001", description="Bone controlling the outer part of the left eyebrow")
    brow_l2: bpy.props.StringProperty(name = "Brow - Left (middle)", default = "brow.T.L.002", description="Bone controlling the middle part of the left eyebrow")
    brow_l3: bpy.props.StringProperty(name = "Brow - Left (inner)", default = "brow.T.L.003", description="Bone controlling the inner part of the left eyebrow")
    brow_r3: bpy.props.StringProperty(name = "Brow - Right (inner)", default = "brow.T.R.003", description="Bone controlling the inner part of the right eyebrow")
    brow_r2: bpy.props.StringProperty(name = "Brow - Right (middle)", default = "brow.T.R.002", description="Bone controlling the middle part of the right eyebrow")
    brow_r1: bpy.props.StringProperty(name = "Brow - Right (outer)", default = "brow.T.R.001", description="Bone controlling the outer part of the right eyebrow")

    #jaw bone name input
    jaw: bpy.props.StringProperty(name = "Jaw", default = "jaw_master", description="Bone controlling the jaw")

    #nose bone name input
    nose_l: bpy.props.StringProperty(name = "Nose - Left", default = "nose.L.001", description="Bone controlling the left side of the nose")
    nose_m: bpy.props.StringProperty(name = "Nose - Tip", default = "nose.002", description="Bone controlling the tip of the nose")
    nose_r: bpy.props.StringProperty(name = "Nose - Right", default = "nose.R.001", description="Bone controlling the right side of the nose")

    #eye bone name input
    eye_l: bpy.props.StringProperty(name = "Eye - Left (outer corner)", default = "lid.B.L", description="Bone controlling the outer corner of left eye")
    eye_tl: bpy.props.StringProperty(name = "Eye - Top left", default = "lid.T.L.002", description="Bone controlling the upper lid of the left eye")
    eye_bl: bpy.props.StringProperty(name = "Eye - Bottom left", default = "lid.B.L.002", description="Bone controlling the lower lid of the left eye")

    eye_r: bpy.props.StringProperty(name = "Eye - Right (outer corner)", default = "lid.B.R", description="Bone controlling the outer corner of right eye")
    eye_tr: bpy.props.StringProperty(name = "Eye - Top right", default = "lid.T.R.002", description="Bone controlling the upper lid of the right eye")
    eye_br: bpy.props.StringProperty(name = "Eye - Bottom right", default = "lid.B.R.002", description="Bone controlling the lower lid of the right eye")

    #input for type of keyframe thats inserted
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

    #bools for rigify show options
    show_names: bpy.props.BoolProperty(name = "Show names", default = False, update=update_bool, description="Show names of bones")
    show_shapes: bpy.props.BoolProperty(name = "Show shapes", default = True, update=update_bool, description="Show shapes of the bones")
    show_col: bpy.props.BoolProperty(name = "Show group colors", default = True, update=update_bool, description="Show group colors of bones")
    show_front: bpy.props.BoolProperty(name = "Show in front", default = True,update=update_bool, description="Show bones in front of their assigned mesh")
    show_axes: bpy.props.BoolProperty(name = "Show axes", default = False, update = update_bool, description="Show local axes of bones")
    #float pre offset osi ktore sa zobrazuju
    axes_offset: bpy.props.FloatProperty(name = "Offset: ", default = 0, precision=2, min = 0, max = 1, update=update_bool, description="Change position of the bone axes. Increasing value moves them to the tip of the bone, decreasing moves them towards the head.")
    

class ArmtInput_PT_armt(bpy.types.Panel):
    bl_label = "Armature"
    bl_idname = "ARMT_PT_armt_input"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Addon"

    #panel pre input mien kosti

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
    bl_label = "Control"
    bl_idname = "MAINPANEL_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Addon"
    bl_options = {"DEFAULT_CLOSED"}
    
    #panel pre spustanie videa kalibracie a vkladanie keyframeov

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
    bl_label = "Rigify"
    bl_idname = "RGF_PT_rigify_opt"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Addon"
    bl_options = {"DEFAULT_CLOSED"}
    
    #panel pre hlavne rigify moznosti 

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
    bl_idname = "calibration.button"
    bl_label = "Calibrate"
    bl_description = "Start calibration"

    def execute(self, context):
        calibrate(context)
        return {'FINISHED'} 


class StartButton(bpy.types.Operator):
    bl_idname = "start.button"
    bl_label = "Start"
    bl_description = "Start capture"

    #ked je stlaceny, precita sa kostra z inputu, inicializuju sa pozicie kosti, kostry atd a vratia sa kosti do povodnej polohy
    #na zaver sa zacne modal timer   
    def execute(self, context):
        global isRunning
        global armt
        global pose_bones
        global error_message

        isRunning = True #zacina sa video
        armt = bpy.context.scene.my_collection_objects #precitame kostru z inputu pouzivatela
        if armt is None: 
            self.report({"ERROR"}, "Armature has not been set. Please choose your armature.")
            return {"CANCELLED"}
        
        pose_bones.clear() 
        init_bone_positions() #inicializujeme pozicie kosti ktore potrebujeme
        if len(error_message) != 0:
            self.report({"ERROR"}, error_message)
            error_message = ''
            return {"CANCELLED"}
        #if len(pose_bones) == 0:
            #return{'CANCELLED'}

        #reset pozicie do base pose 
        bpy.ops.pose.select_all(action = 'SELECT')
        bpy.ops.pose.user_transforms_clear() 
        bpy.ops.pose.select_all(action='DESELECT')
        bpy.ops.wm.modal_timer_operator() #spustenie modal timeru
        return {'FINISHED'}
    
class EndButton(bpy.types.Operator):
    bl_idname = "end.button"
    bl_label = "End"
    bl_description = "End capture. Alternatively press ESC"
    
    #modal timer sa vypne, kosti sa vratia do povodnej polohy
    def execute(self, context):
        global isRunning
        isRunning = False
        bpy.ops.pose.select_all(action = 'SELECT')
        bpy.ops.pose.user_transforms_clear() 
        bpy.ops.pose.select_all(action='DESELECT')
        return {'FINISHED'}
    
class SingleKeyframeButton(bpy.types.Operator):
    bl_idname = "singlekeyframe.button"
    bl_label = "Single Keyframe"
    bl_description = "Add a single keyframe of type chosen in Type"

    #vlozi sa jeden keyframe na snimku kde ma pouzivatel ukazovatel, typ, ktory si vybral, pre vsetky kosti
    def execute(self, context):
        global kf_type
        bpy.ops.pose.select_all(action='SELECT')
        kf_type = bpy.context.scene.my_tool.keyframe_enum #precitanie, ktory typ keyframeu je vybrany
        bpy.ops.anim.keyframe_insert_by_name(type=kf_type) #vlozenie keyframeu
        bpy.context.scene.frame_current += 1 #posunutie o frame
        bpy.ops.pose.select_all(action='DESELECT')
        
        return {'FINISHED'} 

class ContKeyframeButton(bpy.types.Operator):
    bl_idname = "contkeyframe.button"
    bl_label = "Continuous Keyframes"
    bl_description = "Record movement animation in real time"

    #spusti sa vkladanie keyframov pomocou zmeny bool premennej isRecording
    def execute(self, context):
        global isRecording
        global kf_type
        kf_type = bpy.context.scene.my_tool.keyframe_enum 
        if isRecording:
            isRecording = False

            if len(rec_animation) != 0:
                create_anim()
            return {'FINISHED'}
        
        isRecording = True
        return {'FINISHED'}
    

class RigifyFaceButton(bpy.types.Operator):
    bl_idname = "rigifyface.button"
    bl_label = "Face layer"
    bl_description = "Armature face layer visibility"

    #vrstva face vypinanie a zapinanie
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
    bl_idname = "rigifyprimface.button"
    bl_label = "Primary face layer"
    bl_description = "Armature primary face layer visibilty"

    #vrstva primary face vypinanie a zapinanie
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
    bl_idname = "rigifysecface.button"
    bl_label = "Secondary face layer"
    bl_description = "Armature secondary face layer visibilty"

    #vrstva secondary face vypinanie a zapinanie
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
    bl_idname = "rigifytorso.button"
    bl_label = "Torso layer"
    bl_description = "Armature torso layer visibilty"

    #vrstva torso vypinanie a zapinanie
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

def menu_func(self, context):
    self.layout.operator(ModalTimerOperator.bl_idname, text=ModalTimerOperator.bl_label)

def register():
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
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.my_tool
    bpy.types.VIEW3D_MT_view.remove(menu_func)
    del bpy.types.Scene.my_collection
    del bpy.types.Collection.my_collection_objects
    
#if __name__ == "__main__":
register()

