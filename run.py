from UI import MainUI
from cnn import CNN
from knn import KNN
import os
modeltype = "CNN"
if modeltype == "KNN":
    if os.path.exists("knn.sav"):
        pass
    else:
        print("Saved KNN Classifier not found....")
        print("Downloading MNIST Data, training KNN CLassifier and saving the model..........")
        print("Kindly wait for a few minutes..........")
        knnobj = KNN(3)
        knnobj.skl_knn()
else:
    if os.path.exists("cnn.hdf5"):
        pass
    else:
        print("cnn.hdf5 not found...")
        print("Loading MNIST Data, training CNN and saving the model..........")
        print("Kindly wait a few minutes..........")
        cnnobj = CNN()
        cnnobj.build_and_compile_model()
        cnnobj.train_and_evaluate_model()
        cnnobj.save_model()
MainUIobj = MainUI(modeltype)
MainUIobj.mainloop()
MainUIobj.cleanup()