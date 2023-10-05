import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import CNNTextClassifier
from preprocess import Preprocess
import torch.optim as optim
import torch.nn.functional as F
from parser_param import parameter_parser
import numpy as np

SEED = 2019
torch.manual_seed(SEED)

class DataMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class Train:
    def __init__(self, args):
        self.__init_data__(args)
        if torch.cuda.is_available():
            self.device = "cuda:0"
            print("Run on GPU")
        else:
            self.device = "cpu"
            print("Run on CPU")
        self.args = args
        self.batch_size = args.batch_size
        self.model = CNNTextClassifier(args)
        self.model.to(self.device)
    
    def __init_data__(self, args):

        self.preprocess = Preprocess(args)
        self.preprocess.load_data()
        self.preprocess.Tokenization()

        raw_x_train = self.preprocess.X_train
        raw_x_test = self.preprocess.X_test

        self.x_train = self.preprocess.sequence_to_token(raw_x_train)
        self.x_test = self.preprocess.sequence_to_token(raw_x_test)

        self.y_train = self.preprocess.Y_train
        self.y_test = self.preprocess.Y_test

    def train(self):
        training_set = DataMapper(self.x_train, self.y_train)
        test_set = DataMapper(self.x_test, self.y_test)

        self.load_train = DataLoader(training_set, batch_size=self.batch_size)
        self.load_test = DataLoader(test_set, batch_size=self.batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        print(self.model)
      
        for epoch in range(args.epochs):
            prediction = []

            self.model.train()

            for x_batch, y_batch in self.load_train:
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.FloatTensor).unsqueeze(1)
                
                x = x.to(self.device)
                
                y = y.to(self.device)

                y_pred = self.model(x)
                # print(np.shape(y))
               
                loss = F.binary_cross_entropy(y_pred, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                prediction += list(y_pred.cpu().squeeze().detach().numpy())
            # print(prediction[0])
        test_prediction = self.evaluation()
        
        torch.save(self.model.state_dict(), "model/model_CNN.pt")
        train_accuracy = self.calculate_accuracy(self.y_train, prediction)
        test_accuracy = self.calculate_accuracy(self.y_test, test_prediction)
        print("Epoch : %.5f, Loss : %.5f, Train accuracy : %.5f, Loss accuracy : %.5f" % (epoch + 1, loss.item(), train_accuracy, test_accuracy))

    def evaluation(self):

        prediction = []
        self.model.eval()

        with torch.no_grad():
            for x_batch, y_batch in self.load_test:
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.FloatTensor)
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)

                prediction += list(y_pred.cpu().squeeze().detach().numpy())
       
        return prediction

    @staticmethod
    def calculate_accuracy(ground_true, predictions):

        true_positive = 0
        true_negative = 0

        for true, pred in zip(ground_true, predictions):
            if pred >= 0.5 and (true == 1):
                true_positive += 1
            elif pred < 0.5 and true == 0:
                true_negative += 1
            else:
                pass
        return (true_positive + true_negative) / len(ground_true) 

if __name__ == "__main__":
    args = parameter_parser()
    excute = Train(args)
    excute.train()