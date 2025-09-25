
# main.py
import logging
from logger_setup import setup_logging
import model_v0
import model_v1
import model_v2
import model_v3
import summarizer
import io
import contextlib
import train_test


class get_model:
    def __init__(self,device=None, model_version=0):
        self.device = device if device else self.get_device()
        self.model_version = model_version
        self.model_obj = self.get_model(self.model_version)
        self.model_config = self.get_config(self.model_version)

    def get_device(self):
        return train_test.torch.device("cuda" if train_test.torch.cuda.is_available() else "cpu")

    def get_model(self, model_version=0):
        if model_version == 0:
            return model_v0.Net().to(self.device)
        elif model_version == 1:
            return model_v1.Net().to(self.device)
        elif model_version == 2:
            return model_v2.Net().to(self.device)
        elif model_version == 3:
            return model_v3.Net().to(self.device)
        else:
            raise ValueError(f"Unknown model version: {model_version}")
        
    def get_config(self, model_version=0):
        if model_version == 0:
            config = model_v0.set_config_v0()
        elif model_version == 1:
            config = model_v1.set_config_v1()
        elif model_version == 2:
            config = model_v2.set_config_v2()
        elif model_version == 3:
            config = model_v3.set_config_v3()
        else:
            raise ValueError(f"Unknown model version: {model_version}")
        return config.setup(self.model_obj)

def main_i(i, params_check=1):
    logging.info(f"Setting up for model version: {i}")
    model = get_model(model_version=i)
    # Capture printed summary into logs
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        summarizer.summary(model.model_obj, input_size=(1, 28, 28))
        summary_text = buf.getvalue().strip()
    if summary_text:
        logging.info("\n" + summary_text)
    train_test_instance = train_test.train_test_model(model.model_obj,
                                                      model.device, 
                                                      model.model_config.data_setup_instance.train_loader,
                                                      model.model_config.data_setup_instance.test_loader,
                                                      model.model_config.criterion,
                                                      model.model_config.optimizer,
                                                      model.model_config.scheduler,
                                                      model.model_config.epochs)
    if (params_check == 0):
        train_test_instance.run_epoch()
    else:
        pass
    #train_test_instance.plot_results()
    # Capture printed model checks into logs
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        summarizer.model_checks(model.model_obj)
        checks_text = buf.getvalue().strip()
    if checks_text:
        logging.info("\n" + checks_text)

def main():
    # Initialize logging only in the main process
    setup_logging(log_to_file=True)
    set_versions = input("Enter model versions or leave blank for all versions one by one: ")
    params_check = int(input("Enter 1 for params check only, 0 for full training/testing: "))
    if set_versions == "":
        for model_version in range(4): 
            main_i(model_version, params_check=params_check)
    else:
        main_i(int(set_versions), params_check=params_check)

if __name__ == "__main__":
    main()
