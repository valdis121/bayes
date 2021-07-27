import bayes
from tkinter import *
from tkinter import messagebox
from datetime import datetime
from datetime import time
Data=bayes.Dataset()
model = bayes.Bayes()
X_train = 0
X_test = 0
y_train = 0
y_test = 0
Y1 = 0

def LoadData():
    try:
        Data.load(pathData.get())
    except:
        messagebox.showerror('Error', 'File does not exist!')
        pathData.delete(0, 'end')
        return
    all = Data.get_train_test()
    global X_train
    X_train = all[0]
    global X_test
    X_test = all[1]
    global y_train
    y_train = all[2]
    global y_test
    y_test = all[3]
    global Y1
    Y1 = Data.get_names_of_class()


    messagebox.showinfo('Message', 'Data loaded successfully!')
    pathData.delete(0, 'end')
def Train():
    try:
        model.fit(X_train, y_train)
    except :
        messagebox.showerror('Error', 'Check your dataset!')
        return
    messagebox.showinfo('Message', 'Training ended successfully!')
def SaveModel():
    global model
    model.save(saveloadPath.get())
    messagebox.showinfo('Message', 'Model saved successfully')
def LoadModel():
    global model
    try:
        model.load(saveloadPath.get())
    except:
        messagebox.showerror('Error', 'File does not exist!')
        return

    messagebox.showinfo('Message', 'Model loaded successfully')
def TestModel():
    try:
        Accuracy=round(model.test(X_test,y_test)*100,2)
    except:
        messagebox.showerror('Error', 'Check your dataset and model!')
        return
    y_pred = [model.predict(x) for x in X_test]

    from sklearn.metrics import classification_report, confusion_matrix
    matrix=confusion_matrix(y_test, y_pred)
    report=classification_report(y_test, y_pred)
    file = open("logs//{}-{}-{}log.txt".format(datetime.now().hour,datetime.now().minute,datetime.now().second), "w")
    file.write(str(datetime.now())+'\n')
    file.write("\nClassification report:\n")
    file.write(report+"\n")
    file.write("Confusion matrix:\n")
    for row in matrix:
        file.write(str(row))
        file.write('\n')
    file.close()
    messagebox.showinfo('Message', 'Accuracy of the model is {}%. Log file with additional information about testing is saved in the log folder'.format(Accuracy))
def Predict():
    data=input.get().split(",")
    len(data[:15])
    try:
        result=model.predict(data[:14])
    except :
        messagebox.showerror('Error', 'Check your data and model!')
        return
    if result == 0:
        result="edible"
    else:
        result="poisonous"
    messagebox.showinfo('Message', 'Result of prediction: {}'.format(result))



window = Tk()

window.title("Classification of mushrooms")
lbl = Label(window, text="Bayes Classifier",padx=60,pady=10)
lbl.grid(column=1, row=0)
datalbl = Label(window, text="Enter dataset path",padx=20)
datalbl.grid(column=0, row=1,pady=(30,0),sticky = S)
pathData=Entry(window,width=20)
pathData.grid(column=0,row=2,padx=20,pady=10)
dataBtn=Button(window,width=10,text="Load",command=LoadData)
dataBtn.grid(column=1,row=2,pady=10,sticky = W)
trainBtn=Button(window,text="Train", width=10, height=2,command=Train)
trainBtn.grid(column=0,row=4, sticky=SW,  padx=20, pady=10)
filelbl=Label(window,text="Enter file name",width=20)
saveloadPath=Entry(window,width=20)
savebtn=Button(window,width=15, height=1, text="Save model",command=SaveModel)
loadbtn=Button(window,width=15, height=1, text="Load model",command=LoadModel)
filelbl.grid(column=0,row=3, sticky=NE,  padx=20)
saveloadPath.grid(column=0,row=3, sticky=E,  padx=20)
savebtn.grid(column=1,row=3, sticky=NW)
loadbtn.grid(column=1,row=3, sticky=SW,pady=30)
testbtn=Button(window,text="Test", width=10, height=2,command=TestModel)
testbtn.grid(column=3,row=4, sticky=SE,  padx=20,pady=10)
inputlbl=Label(window,text="Enter the data",width=20)
inputlbl.grid(column=3,row=1, sticky=W,pady=(30,0), padx=20)
input=Entry(window,width=20)
input.grid(column=3,row=2,padx=20,pady=10)
classify=Button(window,text="Classify", width=19, height=2,command=Predict)
classify.grid(column=3,row=3, sticky=NW,  padx=20)



window.mainloop()
