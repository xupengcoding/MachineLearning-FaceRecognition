import os
import cv2
import dlib
import cv2
import cv2.cv as cv
import numpy as np
import sys
from skimage import io
import pickle

def cPickle_output(vars, file_name):
    import cPickle
    f = open(file_name, 'wb')
    cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def output_data(vector_vars, vector_folder, batch_size=1000):
    if not vector_folder.endswith('/'):
        vector_folder += '/'
    if not os.path.exists(vector_folder):
        os.mkdir(vector_folder)
    x, y = vector_vars
    n_batch = len(x) / batch_size
    for i in range(n_batch):
        file_name = vector_folder + str(i) + '.pkl'
        batch_x = x[ i*batch_size: (i+1)*batch_size]
        batch_y = y[ i*batch_size: (i+1)*batch_size]
        cPickle_output((batch_x, batch_y), file_name)
    if n_batch * batch_size < len(x):
        batch_x = x[n_batch*batch_size: ]
        batch_y = y[n_batch*batch_size: ]
        file_name = vector_folder + str(n_batch) + '.pkl'
        cPickle_output((batch_x, batch_y), file_name)


def scandir(startdir, ans, last_dir):
    #os.chdir(startdir)
    if not startdir.endswith(os.sep):
        startdir += os.sep
    childlist = os.listdir(startdir)
    for obj in childlist:
        full_path_obj =startdir + obj
        if os.path.isdir(full_path_obj):
            scandir(full_path_obj, ans, last_dir)
            last_dir.append(os.getcwd() + os.sep + full_path_obj)
        else:
            ans.append(full_path_obj)

    #os.chdir(os.pardir)
    return  ans, last_dir

if __name__ == "__main__":
    dir = sys.argv[1]
    #dir = "test"
    f = open(dir+"_facelog.txt", 'w')
    face_pickefile = dir + "_faceIndexPkl"
    img_vec = []
    dir_vec = []
    scandir(dir, img_vec, dir_vec)
    #create dir
    dir_vec = list(set(dir_vec))
    for obj_dir in dir_vec:
        new_dir = obj_dir.replace(dir, dir+"_face")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    detector = dlib.get_frontal_face_detector()
    face_counter = 0
    face_id_vec = []
    face_save_path_vec = []
    for img_path in img_vec:
        img_flag = 1
        try:
            img = cv2.imread(img_path)
        except Exception:
            print img_path + " can not open as a image"
            img_flag = 0

        if img_flag == 0:
            continue
        #img1 = io.imread(img_path)
        dets = detector(img, 1)
        for i, d in enumerate(dets):
            if d.top() < 0 or d.left() < 0 or d.right() < 0 or d.bottom() < 0:
                continue
            img_face = img[d.top():d.bottom(),d.left():d.right(),:]
            #cv2.imshow("test", img_face
            #cv2.waitKey(0)
            suffix = "_"+str(i)+".jpg"
            face_save_path = img_path.replace(".jpg", suffix)
            face_save_path = face_save_path.replace(".JPG", suffix)
            face_save_path = face_save_path.replace(dir, dir+"_face")
            img_face = cv2.resize(img_face, (224, 224))
            cv2.imwrite(face_save_path, img_face)
            f.writelines(str(face_counter) + " " + img_path + " " + str(d.left())+" "+str(d.top())+" "+str(d.right())+" "+str(d.bottom()) + "\n")
            face_id_vec.append(face_counter)
            face_counter += 1
            face_save_path_vec.append(face_save_path)
            if "gallery" in face_save_path:
                break
        print "done"+img_path

    f.close()
    face_id_vec = np.asarray(face_id_vec, dtype='int32')
    face_save_path_vec = np.asarray(face_save_path_vec, dtype="string")
    output_data((face_id_vec, face_save_path_vec), face_pickefile)