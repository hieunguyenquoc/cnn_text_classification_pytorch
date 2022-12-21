import torch
from model import CNNTextClassifier
from preprocess import Preprocess
from parser_param import parameter_parser

class Inference:
    def __init__(self, args):
        self.model = CNNTextClassifier(args)
        self.model = torch.load("model/model_CNN.ckpt")
        self.model.eval()
        self.preprocess = Preprocess(args)
        self.preprocess.load_data()
        self.preprocess.Tokenization()

    def prediction(self, sentence):
        # prediction = []
        sentence_to_pred = self.preprocess.sequence_to_token(sentence)
        sentence_tensor = torch.from_numpy(sentence_to_pred)
        sentence_pred = sentence_tensor.type(torch.LongTensor)
        with torch.no_grad():
            y = self.model(sentence_pred)
            # prediction += list(y.detach().numpy())
        
        return y

if __name__ == "__main__":
    args = parameter_parser()
    output = Inference(args)
    sentence = "Burners follow @ablaze"
    print(output.prediction([sentence]))
    if output.prediction([sentence]) <= 0.5:
        print('Score : {}, Prediction : positive'.format(output.prediction([sentence])))
    else:
        print('Score : {}, Prediction : negative'.format(output.prediction([sentence])))
            

        