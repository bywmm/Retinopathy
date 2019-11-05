from model import Inceptionv3WithAttention
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--resume", default=False, type=bool, help="resume training or not")
    parser.add_argument("--with_att", default=True, type=bool, help='with attention or not')
    parser.add_argument("--mini_data", default=True, type=bool, help='use mini data or not')
    parser.add_argument("--batch_size", default=48, type=int, help='batch size for training')
    parser.add_argument("--init_epoch", default=20, type=int, help="number of init epoch for resume")
    parser.add_argument("--epochs", default=20, type=int, help="number of traning epochs")
    parser.add_argument("--out_name", help='out key words')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    model = Inceptionv3WithAttention(num_classes=5)
    if args.resume:
        model_json = 'checkpoints/'+str(args.out_name)+'/net_arch.json'
        model_weights = 'checkpoints/'+str(args.out_name)+'/weights_epoch'+str(args.init_epoch-1)+'.h5'
        model.resume_train(args.batch_size, model_json, model_weights, args.init_epoch,
                           args.epochs, args.out_name, args.mini_data)
    else:
        # train
        model.build_model(with_att=True, show=True)
        print(args)
        print("args.mini_data:", args.mini_data)
        model.train(args.batch_size, args.epochs, args.out_name, args.mini_data)
