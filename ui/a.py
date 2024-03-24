from customtkinter import *
from PIL import Image


app = CTk()
app.geometry("900x700")


def classify():
    frame=CTkFrame(master=app,fg_color="green",border_color="black",border_width=2)
    frame.place(relx = 0.8, rely = 0.1,anchor="center")
    txt=CTkLabel(master=frame,text="Car")
    txt.pack(anchor="s",expand=True,pady=3,padx=3)


def selectfile():
    filename=filedialog.askopenfilename()
    print(filename)
    img=Image.open(filename)
    image=CTkImage(light_image=img,dark_image=img,size=(300,400))
    imLabel=CTkLabel(app,text="",image=image)
    imLabel.place(relx = 0.3, rely = 0.3,anchor="center")


button_to_select = CTkButton(master=app, text = "Choose file", fg_color = "blue", command = selectfile)
button_to_select.pack(padx = 5, pady = 5)
button_to_select.place(relx = 0.5, rely = 0.9,anchor="center")

classify_button=CTkButton(master=app,text="Classify",fg_color="green",command=classify)
classify_button.place(relx=0.7,rely=0.7,anchor="center")

app.mainloop()

