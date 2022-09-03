window = tkinter.Tk()
window.title("Face and Facemask Recognition Based Attendance System")
window.geometry("1280x720")
window.resizable(True,True)
window.configure(background='#355454')

#Help menubar----------------------------------------------
menubar=Menu(window)
help=Menu(menubar,tearoff=0)
help.add_command(label="Change Password!",command=change_pass)
help.add_command(label="Contact Us",command=contact)
help.add_separator()
help.add_command(label="Exit",command=on_closing)
menubar.add_cascade(label="Help",menu=help)

# add ABOUT label to menubar-------------------------------
menubar.add_command(label="About",command=about)

#This line will attach our menu to window
window.config(menu=menubar)

#main window------------------------------------------------
message3 = tkinter.Label(window, text="Face and Facemask Recognition Based Attendance System" ,fg="white",bg="#355454" ,width=60 ,height=1,font=('times', 29, ' bold '))
message3.place(x=10, y=10,relwidth=1)

#frames-------------------------------------------------
frame1 = tkinter.Frame(window, bg="white")
frame1.place(relx=0.11, rely=0.15, relwidth=0.39, relheight=0.80)

frame2 = tkinter.Frame(window, bg="white")
frame2.place(relx=0.51, rely=0.15, relwidth=0.39, relheight=0.80)

#frame_headder
fr_head1 = tkinter.Label(frame1, text="Register New Student", fg="white",bg="black" ,font=('times', 17, ' bold ') )
fr_head1.place(x=0,y=0,relwidth=1)

fr_head2 = tkinter.Label(frame2, text="Student's attendance", fg="white",bg="black" ,font=('times', 17, ' bold ') )
fr_head2.place(x=0,y=0,relwidth=1)

#registretion frame
lbl = tkinter.Label(frame1, text="Enter ID",width=20  ,height=1  ,fg="black"  ,bg="white" ,font=('times', 17, ' bold ') )
lbl.place(x=0, y=55)

txt = tkinter.Entry(frame1,width=32 ,fg="black",bg="#e1f2f2",highlightcolor="#00aeff",highlightthickness=3,font=('times', 15, ' bold '))
txt.place(x=55, y=88,relwidth=0.75)

lbl2 = tkinter.Label(frame1, text="Enter Name",width=20  ,fg="black"  ,bg="white" ,font=('times', 17, ' bold '))
lbl2.place(x=0, y=140)

txt2 = tkinter.Entry(frame1,width=32 ,fg="black",bg="#e1f2f2",highlightcolor="#00aeff",highlightthickness=3,font=('times', 15, ' bold ')  )
txt2.place(x=55, y=173,relwidth=0.75)

message0=tkinter.Label(frame1,text="Follow the steps...",bg="white" ,fg="black"  ,width=39 ,height=1,font=('times', 16, ' bold '))
message0.place(x=7,y=275)

message1 = tkinter.Label(frame1, text="1)Take Images  ===> 2)Save Profile" ,bg="white" ,fg="black"  ,width=39 ,height=1, activebackground = "yellow" ,font=('times', 15, ' bold '))
message1.place(x=7, y=300)

message = tkinter.Label(frame1, text="" ,bg="white" ,fg="black"  ,width=39,height=1, activebackground = "yellow" ,font=('times', 16, ' bold '))
message.place(x=7, y=500)
#Attendance frame
lbl3 = tkinter.Label(frame2, text="Attendance Table",width=20  ,fg="black"  ,bg="white"  ,height=1 ,font=('times', 17, ' bold '))
lbl3.place(x=100, y=115)

#Display total registration----------
res=0
exists = os.path.isfile("StudentDetails/StudentDetails.csv")
if exists:
    with open("StudentDetails/StudentDetails.csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text='Total Registrations : '+str(res))

#BUTTONS----------------------------------------------

clearButton = tkinter.Button(frame1, text="Clear", command=clear, fg="white", bg="#13059c", width=11, activebackground = "white", font=('times', 12, ' bold '))
clearButton.place(x=55, y=230,relwidth=0.29)

takeImg = tkinter.Button(frame1, text="Take Images", command=TakeImages, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
takeImg.place(x=30, y=350,relwidth=0.89)

trainImg = tkinter.Button(frame1, text="Save Profile", command=psw, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
trainImg.place(x=30, y=430,relwidth=0.89)

trackImg = tkinter.Button(frame2, text="Take Attendance", command=TrackImages, fg="black", bg="#00aeff", height=1, activebackground = "white" ,font=('times', 16, ' bold '))
trackImg.place(x=30,y=60,relwidth=0.89)

quitWindow = tkinter.Button(frame2, text="Quit", command=window.destroy, fg="white", bg="#13059c", width=35, height=1, activebackground = "white", font=('times', 16, ' bold '))
quitWindow.place(x=30, y=450,relwidth=0.89)



#Attandance table----------------------------
style = ttk.Style()
style.configure("mystyle.Treeview", highlightthickness=0, bd=0, font=('Calibri', 11)) # Modify the font of the body
style.configure("mystyle.Treeview.Heading",font=('times', 13,'bold')) # Modify the font of the headings
style.layout("mystyle.Treeview", [('mystyle.Treeview.treearea', {'sticky': 'nswe'})]) # Remove the borders
tb= ttk.Treeview(frame2,height =13,columns = ('name','date','time'),style="mystyle.Treeview")
tb.column('#0',width=82)
tb.column('name',width=130)
tb.column('date',width=133)
tb.column('time',width=133)
tb.grid(row=2,column=0,padx=(0,0),pady=(150,0),columnspan=4)
tb.heading('#0',text ='ID')
tb.heading('name',text ='NAME')
tb.heading('date',text ='DATE')
tb.heading('time',text ='TIME')

#SCROLLBAR--------------------------------------------------

scroll=ttk.Scrollbar(frame2,orient='vertical',command=tb.yview)
scroll.grid(row=2,column=4,padx=(0,100),pady=(150,0),sticky='ns')
tb.configure(yscrollcommand=scroll.set)

#closing lines------------------------------------------------
window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()