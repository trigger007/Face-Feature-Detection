import cv2
from tkinter import *
import os
import numpy as np
from PIL import Image,ImageTk
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense,Flatten
from keras.models import Sequential, load_model


if not os.path.isdir('DATA'):
    os.mkdir('DATA')
    os.mkdir('DATA/train')
    os.mkdir('DATA/test')
    

root = Tk()
root.title("Face-Recognition")
img_frame = LabelFrame(root, text = "Developed by Akarsh", width = 600, height = 500)
img_frame.grid(row=0,column=0,rowspan = 2)

label_frame = Frame(root, width = 200, height = 250)
label_frame.grid(row=1,column = 1)

label_frame_top = Frame(root, width = 200, height = 250)
label_frame_top.grid(row=0,column=1)

img_label = Label(img_frame)
img_label.grid(row=0,column = 0)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

i = 1
but = 0
shift = 0
text = 0
state = NORMAL
my_dict = {}

def face_recog(img):
    global my_dict
    try:
        if not os.path.isfile('face_scan.h5'):
            print("Please train your model first!")
        else:
            new_model = load_model('face_scan.h5')
            img = cv2.resize(img,(32,32),3)
            img = np.expand_dims(img,axis=0)
            img = img / 255

            prediction = new_model.predict_classes(img)
            prediction = prediction[0]
            accuracy = new_model.predict(img)
            #print(accuracy)
            val = round(accuracy[0][0],2)
            if len(my_dict) == 2:
                for key, value in my_dict.items():
                    if prediction == value and (val <= 0.10 or val >= 0.60): 
                        return key

                else:
                    return 'Unknown'

            else:
                for key, value in my_dict.items():
                    if prediction == value and accuracy[0][prediction] > 0.5: 
                        return key

                else:
                    return 'Unknown'
    except:
        print("Error while Resizing")



def train_model():
    global but, shift
    if len(os.listdir('DATA/train')) != 0:
        
        folder = []
        for names in os.listdir('DATA/train'):
            folder.append(names)
        
        print(folder)
        

        if len(folder) ==2:
            cat = 'binary'
            folder = 1
            loss = 'binary_crossentropy'
            metric = ['accuracy']
            act = 'sigmoid'
        else:
            cat = 'categorical'
            folder = len(folder)
            loss = 'categorical_crossentropy'
            metric = ['accuracy']
            act = 'softmax'
            
        print("The length of directory is:",folder)
        
        print(cat,folder,loss,metric,act)
    
        datagen = ImageDataGenerator(rescale = 1./255,
                                    horizontal_flip = True,
                                    shear_range = 0.3,
                                    zoom_range = 0.3,
                                    width_shift_range = 0.3,
                                    height_shift_range = 0.3)

        trained_image = datagen.flow_from_directory('DATA/train',
                                                   target_size = (32,32),
                                                   class_mode = cat)

        test_datagen = ImageDataGenerator(rescale = 1./255)

        test_image = test_datagen.flow_from_directory('DATA/test',
                                                     target_size = (32,32),
                                                     class_mode = cat)
        
        model = Sequential()
        conv_base = VGG16(weights='imagenet', include_top = False, input_shape = (32,32,3))
        #conv_base.trainable = False
        model.add(conv_base)
        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dense(folder,activation = act))
        
        model.compile(optimizer= 'adam', loss = loss, metrics = metric)
        model.fit(trained_image, epochs = 10, validation_data = test_image, validation_steps = 1)
        
        model.save('face_scan.h5')
        
        shift = 0
        but = 0
        button()
    
    elif len(os.listdir('DATA/train')) == 0:
        shift = 0
        but = 0
        button()
        
        


def add():
    global but
    if but == 0:
        but = 1
    else:
        but = 0
        
    button()
    
def sub():
    global but, shift
    but = 0
    shift = 0
    button()
        
def pred():
    global shift, but, my_dict
    datagen = ImageDataGenerator(rescale = 1./255,
                                    horizontal_flip = True,
                                    shear_range = 0.3,
                                    zoom_range = 0.3,
                                    width_shift_range = 0.3,
                                    height_shift_range = 0.3)
    trained_image = datagen.flow_from_directory('DATA/train',
                                                   target_size = (32,32))
    
    my_dict = dict(trained_image.class_indices)
    
    
    
    
    shift = 2
    but = 2
    button()


def train():
    global shift
    shift = 3
    
def scanner():
    global i, shift, text, but, state
    
    if os.path.isfile('DATA/train/.DS_Store'):
        os.remove('DATA/train/.DS_Store')
    if os.path.isfile('DATA/test/.DS_Store'):
        os.remove('DATA/test/.DS_Store')
    
    if shift == 0:
        try:
            state = NORMAL
            success, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img,1)
            face_rects = face_cascade.detectMultiScale(img,scaleFactor=1.2,minNeighbors =5)
            img_tk = ImageTk.PhotoImage(Image.fromarray(img))
            img_label.imgtk = img_tk
            img_label.configure(image=img_tk)
            img_label.after(10,scanner)
        except:
            print("Webcam error!")
            sys.exit()
    
    elif shift == 1 and text != 0:
        try:
            success, img = cap.read()
            img = cv2.flip(img,1)
            image = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_rects = face_cascade.detectMultiScale(image,scaleFactor=1.2,minNeighbors =5)
            
            if i <= 300:
                for (x,y,w,h) in face_rects:
                    image = image[(y-40):(y+h+30),(x-40):(x+w+20)]
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.imwrite(f"DATA/train/{text}/{i}.png",image)
                    i += 1
                cv2.putText(img,f"Scanning....{i/4}%",(40,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                img_tk = ImageTk.PhotoImage(Image.fromarray(img))
                img_label.imgtk = img_tk
                img_label.configure(image=img_tk)
                img_label.after(10,scanner)
                    
                
            elif i > 300 and i <= 400:
                for (x,y,w,h) in face_rects:
                    image = image[(y-40):(y+h+30),(x-40):(x+w+20)]
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.imwrite(f"DATA/test/{text}/{i}.png",image)
                    i += 1
                cv2.putText(img,f"Scanning....{i/4}%",(40,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                img_tk = ImageTk.PhotoImage(Image.fromarray(img))
                img_label.imgtk = img_tk
                img_label.configure(image=img_tk)
                img_label.after(10,scanner)
                    
            
            else:
                but = 0
                i = 1
                shift = 0
                scanner()
                button()
            

        
        except:
            print("Webcam Error!")
            scanner()
            
    elif shift == 2:
        success, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img,1)
        image = img.copy()
        face_rects = face_cascade.detectMultiScale(image,scaleFactor=1.2,minNeighbors =5)
        for (x,y,w,h) in face_rects:
            image = image[(y-40):(y+h+30),(x-40):(x+w+20)]
            cv2.rectangle(img,(x,y),(x+w,y+h),(160,255,13),2)
            cv2.rectangle(img,(x-1,y-40),(x+w-30,y),(160,255,13),-1)
            cv2.putText(img,f"{face_recog(image)}",(x+5,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        img_label.imgtk = img_tk
        img_label.configure(image=img_tk)
        img_label.after(10,scanner)
        

            
def button():
    global but
    if but == 0:
        e = Entry(label_frame_top, width = 19, state = DISABLED)
        my_label = Label(label_frame_top, text="                    ")
        my_label.grid(row=0, column=0, padx=5, pady=2)
        e.grid(row= 1, column = 0, padx = 5, pady = 2)
            
        top_scan_button = Button(label_frame_top, text = "VERIFY", height = 4, width = 20,state=DISABLED)
        top_scan_button.grid(row=2, column=0, padx=5, pady=2)
        return_button = Button(label_frame_top, text = "RETURN", command = add, height = 4, width = 20, state = DISABLED)
        return_button.grid(row=3, column = 0, padx=5, pady =2)
        scan_button = Button(label_frame, text = "SCAN", command = add, height=4, width=20 )
        scan_button.grid(row = 0, column=0, padx=5,pady=2)
        predict_button = Button(label_frame, text = "PREDICT", command = pred, height=4, width=20)
        predict_button.grid(row=1, column = 0,padx = 5, pady=2)
        train_button = Button(label_frame, text = "TRAIN", command = train_model,height=4, width=20)
        train_button.grid(row=2,column=0,padx=5,pady=2)
        
    elif but == 1:
        e = Entry(label_frame_top, width = 19)
        my_label = Label(label_frame_top, text="NAME:")
        my_label.grid(row=0, column=0, padx=5, pady=2)
        e.grid(row= 1, column = 0, padx = 5, pady = 2)
        def name():
            global text, shift, state
            stat = e.get()
            if not os.path.isdir(f"DATA/train/{stat}") and len(stat) < 15:
                os.mkdir(f"DATA/train/{stat}")
                os.mkdir(f"DATA/test/{stat}")
                e.delete(0,END)
                text = stat
                shift = 1
                state = DISABLED
            else:
                e.delete(0,END)
                e.insert(0,"Name Exists......!")
                text = 0
                shift = 0
                state = NORMAL
            
            button()
            
        top_scan_button = Button(label_frame_top, text = "VERIFY",command = name, height = 4, width = 20, state = state)
        top_scan_button.grid(row=2, column=0, padx=5, pady=2)
        return_button = Button(label_frame_top, text = "RETURN", command = add, height = 4, width = 20, state = state)
        return_button.grid(row=3, column = 0, padx=5, pady =2)
        scan_button = Button(label_frame, text = "SCAN", command = name, height=4, width=20, state= DISABLED )
        scan_button.grid(row = 0, column=0, padx=5,pady=2)
        
        predict_button = Button(label_frame, text = "PREDICT", command = pred, height=4, width=20, state = DISABLED)
        predict_button.grid(row=1, column = 0,padx = 5, pady=2)
        train_button = Button(label_frame, text = "TRAIN", command = scanner,height=4, width=20, state= DISABLED)
        train_button.grid(row=2,column=0,padx=5,pady=2)
        
    elif but == 2:
        e = Entry(label_frame_top, width = 19, state = DISABLED)
        my_label = Label(label_frame_top, text="                    ")
        my_label.grid(row=0, column=0, padx=5, pady=2)
        e.grid(row= 1, column = 0, padx = 5, pady = 2)
            
        top_scan_button = Button(label_frame_top, text = "VERIFY", height = 4, width = 20,state=DISABLED)
        top_scan_button.grid(row=2, column=0, padx=5, pady=2)
        return_button = Button(label_frame_top, text = "RETURN", command = sub, height = 4, width = 20)
        return_button.grid(row=3, column = 0, padx=5, pady =2)
        scan_button = Button(label_frame, text = "SCAN", command = add, height=4, width=20 , state = DISABLED)
        scan_button.grid(row = 0, column=0, padx=5,pady=2)
        predict_button = Button(label_frame, text = "PREDICT", command = pred, height=4, width=20, state = DISABLED)
        predict_button.grid(row=1, column = 0,padx = 5, pady=2)
        train_button = Button(label_frame, text = "TRAIN", command = train_model,height=4, width=20, state = DISABLED)
        train_button.grid(row=2,column=0,padx=5,pady=2)
    
    
        
scanner()
button()

root.mainloop()