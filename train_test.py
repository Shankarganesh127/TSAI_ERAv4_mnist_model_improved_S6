
# train_test.py
from tqdm import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


class train_test_model:
    
    def __init__(self, model, device, train_loader, test_loader):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = self.set_criterion()
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()
        self.train_acc_list = []
        self.test_acc_list = []

    def set_criterion(self):
        return torch.nn.NLLLoss()

    def set_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)

    def set_scheduler(self):
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.1)

    def train(self, model, device, train_loader, optimizer, criterion):
        self.model.train()
        pbar = tqdm(self.train_loader)
        train_loss, correct, processed = 0, 0, 0
        for data, target in pbar:
            # get samples and move to device
            data, target = data.to(self.device), target.to(self.device)
            # Initialize optimizer
            self.optimizer.zero_grad()
            # Prediction
            output = self.model(data)
            # Calculate loss
            loss = self.criterion(output, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            # -----------------------------
            # Accumulate loss and calculate accuracy
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            
            # Update progress bar with current statistics
            pbar.set_description(desc=f"Train Loss={train_loss / processed:.4f} Accuracy={100. * correct / processed:.2f}")

        return 100. * correct / len(self.train_loader.dataset)

    def test(self, model, device, test_loader, criterion):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        acc = 100. * correct / len(self.test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({acc:.2f}%)\n')
        return acc

    def do_training(self):
        return self.train(self.model, self.device, self.train_loader, self.optimizer, self.criterion)

    def do_testing(self):
        return self.test(self.model, self.device, self.test_loader, self.criterion)
    
    def run_epoch(self, epochs=1):
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}")
            train_acc = self.do_training()
            test_acc = self.do_testing()
            self.scheduler.step()
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            
    def plot_results(self):
        plt.plot(self.train_acc_list, label='Train Acc')
        plt.plot(self.test_acc_list, label='Test Acc')
        plt.legend()
        plt.title("Training vs Test Accuracy")
        plt.show()



