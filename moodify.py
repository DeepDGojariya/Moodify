import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tkinter as tk
import random
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

cap = cv2.VideoCapture(0)
cv2.namedWindow('Moodify', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Moodify', 600, 400)
saved_model = load_model("model_2.h5")
opDict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
lst = []
start = False
i = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.rectangle(frame, (50, 50), (450, 450), (255, 201, 214), 2)
    if not start:
        cv2.putText(frame, 'Press S for starting and keep your face inside the box. Q=Quit',
                    (5, 25), fontFace=cv2.FONT_ITALIC, fontScale=0.6, color=(100, 25, 242), thickness=1,
                    lineType=cv2.LINE_AA)
    # Display the resulting frame
    if (cv2.waitKey(60) & 0xFF == ord('s')) or start:
        cv2.putText(frame, 'Collecting Facial Data',
                    (5, 25), fontFace=cv2.FONT_ITALIC, fontScale=0.6, color=(100, 25, 242), thickness=1,
                    lineType=cv2.LINE_AA)
        print("Pressed S")
        i += 1
        if i > 25:
            break

        start = True
        gray = frame[50:450, 50:450]
        gray = cv2.cvtColor(gray, code=cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))
        result = saved_model.predict(gray[np.newaxis, :, :, np.newaxis])
        result = list(result[0])
        img_index = result.index(max(result))
        lst.append(opDict[img_index])
    cv2.imshow('Moodify', frame)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

mood = max(set(lst), key=lst.count)

data1 = {'Emotion': ['Happy', 'Sad', 'Angry', 'Surprise'],
         'Score': [lst.count('Happy'), lst.count('Sad'), lst.count('Angry'), lst.count('Surprise')]
         }
df1 = DataFrame(data1, columns=['Emotion', 'Score'])

song_dict = {
    'Happy': ['Say Yes', 'Valerie', "Let's go crazy", 'Tightrope', 'Walking on Sunshine', 'I got you', 'Uptown Funk',
              'Happy', 'You make me feel like'],
    'Sad': ['Time after Time', 'Your Song', 'True Colours', "Dance with my Father", "Against all odds",
            "I can't tell you why"],
    'Angry': ["You don't owe me", "Dreamin", "Female Robbery", "Bury a Friend", "Without Me", "Holdin On", "Genesis",
              "Hell No"],
    'Surprise': ['Sweden', 'Limit to your Love', 'I wanna be yours', 'Metallica', 'Pink Floyd', 'Raabta']
    }
lst_r = random.sample(song_dict[mood], 5)


frame = tk.Tk()
frame.title('Moodify')
frame.geometry("700x700")

listbox = tk.Listbox(frame, bg="grey", activestyle='dotbox', font="Helvetica", fg="yellow")
listbox.pack(padx=30, pady=30, fill=tk.BOTH, expand=False)
listbox.insert(0,"Here are your Song recommendations.")
listbox.insert(1,"We Predicted that you are {}".format(mood))
listbox.insert(3,"\n")
for i in range(len(lst_r)):
    listbox.insert(i+3, str(i + 1) + '.' + lst_r[i])

figure1 = plt.Figure(figsize=(5, 5), dpi=90)
ax1 = figure1.add_subplot(111)
bar1 = FigureCanvasTkAgg(figure1, frame)
bar1.get_tk_widget().pack(fill=tk.BOTH)
df1 = df1[['Emotion', 'Score']].groupby('Emotion').sum()
df1.plot(kind='barh', legend=True, ax=ax1)
ax1.set_title('Emotion Detector Results(out of 25)')


tk.mainloop()
