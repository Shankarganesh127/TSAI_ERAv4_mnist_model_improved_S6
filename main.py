
# main.py
import model_v0
import summarizer
import train_test
import data_setup

class model_getter:
    def __init__(self,device=None):
        self.device = device if device else self.get_device()
        self.model = self.get_model()

    def get_device(self):
        return train_test.torch.device("cuda" if train_test.torch.cuda.is_available() else "cpu")

    def get_model(self):
        return model_v0.Net().to(self.device)

def main():
    # Initialize model, data loaders, and training/testing framework
    model = model_getter().model
    summarizer.summary(model, input_size=(1, 28, 28))
    if model.device.type == "cuda":
        dataloader_args = dict(batch_size_train=64, batch_size_test=1000, shuffle_train=True, shuffle_test=False, num_workers=2, pin_memory=True) 
    else:
        dataloader_args = dict(batch_size_train=64, batch_size_test=1000, shuffle_train=True, shuffle_test=False)
    data_setup_instance = data_setup.DataSetup(**dataloader_args)
    train_test_instance = train_test.train_test_model(model, model.device, data_setup_instance.train_loader, data_setup_instance.test_loader)
    train_test_instance.run_epoch(epochs=15)
    train_test_instance.plot_results()
    summarizer.model_checks(model)

if __name__ == "__main__":
    main()
