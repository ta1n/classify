from customtkinter import *
from PIL import Image

from predict import *

app = CTk()
app.geometry("900x700")

image_file = '../data/test/Bike/Bike (1070).jpeg'

def selectfile():
    filename=filedialog.askopenfilename()
    print(filename)
    global image_file
    image_file=filename
    img=Image.open(filename)
    image=CTkImage(light_image=img,dark_image=img,size=(300,400))
    imLabel=CTkLabel(app,text="",image=image)
    imLabel.place(relx = 0.3, rely = 0.3,anchor="center")



def classify():
    model = CarBikeClassifier(num_classes=2)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    print(image_file)
    prediction_text=predict_image(model, image_file, device='cpu')

    frame=CTkFrame(master=app,fg_color="green",border_color="black",border_width=2)
    frame.place(relx = 0.8, rely = 0.1,anchor="center")
    txt=CTkLabel(master=frame,text=prediction_text)
    txt.pack(anchor="s",expand=True,pady=3,padx=3)



button_to_select = CTkButton(master=app, text = "Choose file", fg_color = "blue", command = selectfile)
button_to_select.pack(padx = 5, pady = 5)
button_to_select.place(relx = 0.5, rely = 0.9,anchor="center")

classify_button=CTkButton(master=app,text="Classify",fg_color="green",command=classify)
classify_button.place(relx=0.7,rely=0.7,anchor="center")

app.mainloop()

