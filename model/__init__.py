import model.MPIIGaze.LeNet as LeNet

def create_model(model="MPIIGaze"):
    if model == "MPIIGaze":
        model = LeNet.LeNet()
        return model
