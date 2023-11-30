import os
import argparse
from model_structure import UNet
from Model_trainer import Model_trainer

def training_procedure():
    ''' Main function for training a U-Net model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to dataset')
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print('Path does not exist')
        return  
    else:
        model = UNet()
        trainer = Model_trainer(model = model, epochs = 100, set_dir = f'{args.path}*.jpg', learning_rate=0.001)
        trainer.train_model()

if __name__ == "__main__":
    training_procedure ()