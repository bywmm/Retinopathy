from model import Inceptionv3WithAttention
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--batch_size", default=48, type=int, help='batch size for training')
    parser.add_argument("--epochs", type=int, help="number of traning epochs")
    parser.add_argument("--mini_data", default=True, type=bool, help='use mini data or not')
    parser.add_argument("--out_name", help='out key words')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    model = Inceptionv3WithAttention(num_classes=5)

    model_json = 'checkpoints/'+str(args.out_name)+'/net_arch.json'
    model_weights = 'checkpoints/'+str(args.out_name)+'/weights_epoch'+str(args.epochs-1)+'.h5'

    model.load_model(model_json, model_weights)
    model.eval(args.batch_size, args.out_name)