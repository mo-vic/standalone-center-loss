import os
import sys
import argparse
from datetime import datetime

import torch
from torch.backends import cudnn
from tensorboardX import SummaryWriter

from utils.utils import load_dataset, build_model
from utils.utils import train, eval
from utils.logger import Logger

from losses.xentropy_loss import CrossEntropyLoss


def main():
    parser = argparse.ArgumentParser(description="Softmax + Xentropy")

    # Dataset
    parser.add_argument("--dataset", type=str, default="fashion-mnist", choices=["mnist", "fashion-mnist", "cifar-10"])
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers.")
    # Optimization
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--gpu_ids", type=str, default='', help="GPUs for running this script.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for gradient descent.")
    parser.add_argument("--factor", type=float, default=0.2, help="Factor by which the learning rate will be reduced.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Threshold for measuring the new optimum, to only focus on significant changes. ")
    # Model
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet"])
    parser.add_argument("--feat_dim", type=int, default=128, help="Dimension of the feature.")
    # Misc
    parser.add_argument("--log_dir", type=str, default="./run/", help="Where to save the log?")
    parser.add_argument("--log_name", type=str, required=True, help="Name of the log folder.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--eval_freq", type=int, default=1, help="How frequently to evaluate the model?")
    parser.add_argument("--vis", action="store_true", help="Whether to visualize the features?")

    args = parser.parse_args()

    # Check before run.
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    log_dir = os.path.join(args.log_dir, args.log_name)

    # Setting up logger
    log_file = datetime.now().strftime("%Y-%m-%d-%H-%M-%S_{}.log".format(args.dataset))
    sys.stdout = Logger(os.path.join(log_dir, log_file))
    print(args)

    for s in args.gpu_ids:
        try:
            int(s)
        except ValueError as e:
            print("Invalid gpu id:{}".format(s))
            raise ValueError

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)

    if args.gpu_ids:
        if torch.cuda.is_available():
            use_gpu = True
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)
        else:
            use_gpu = False
    else:
        use_gpu = False

    torch.manual_seed(args.seed)

    trainloader, testloader, input_shape, classes = load_dataset(args.dataset, args.batch_size, use_gpu,
                                                                 args.num_workers)
    model = build_model(args.model, input_shape, args.feat_dim, len(classes))
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.factor,
                                                           patience=args.patience, verbose=True,
                                                           threshold=args.threshold)

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    print("Start training...")
    start = datetime.now()
    with SummaryWriter(log_dir) as writer:
        for epoch in range(args.epochs):
            train(model, trainloader, criterion, optimizer, use_gpu, writer, epoch, args.epochs, args.vis,
                  args.feat_dim, classes)

            if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
                eval(model, testloader, criterion, scheduler, use_gpu, writer, epoch, args.epochs, args.vis,
                     args.feat_dim, classes)

    elapsed_time = str(datetime.now() - start)
    print("Finish training. Total elapsed time %s." % elapsed_time)


if __name__ == "__main__":
    main()
