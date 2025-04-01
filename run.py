import matplotlib.pyplot as plt

# 设置 Matplotlib 字体为 SimHei（适用于 Windows）,解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings

warnings.filterwarnings("ignore")
import argparse
import os
from utils import *
from models import *


def get_args():
    parser = argparse.ArgumentParser(description='Model Configuration')

    # 实验配置
    parser.add_argument('--experiment_name', type=str, default='default_experiment', help='Name of the experiment')

    # 数据集配置
    parser.add_argument('--file_path', type=str, default='./data/FJ.csv', help='Path to dataset file')
    parser.add_argument('--history_length', type=int, default=7, help='Historical window size')
    parser.add_argument('--prediction_length', type=int, default=7, help='Number of prediction steps')
    parser.add_argument('--target_var', type=str, default='CO_2', help='Target variable for prediction')
    parser.add_argument('--input_var', type=str, nargs='+',
                        default=['IN_T', 'IN_BT', 'IN_RH', 'CO_2', 'OUT_T', 'OUT_RH', 'WS'],
                        help='List of input variables')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation set')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--hidden_units', type=int, default=32, help='Number of hidden units in the model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    dataset_name = os.path.splitext(os.path.basename(args.file_path))[0]
    save_dir = f'./results1/{args.experiment_name}/{args.file_path.split("/")[-1].split(".")[0]}_HL{args.history_length}_PL{args.prediction_length}_{args.target_var}/'
    os.makedirs(save_dir, exist_ok=True)

    # Read dataset
    print("==> Load dataset ...")
    processor = DataProcessor(file_path=args.file_path, windows=args.history_length,
                              pred_len=args.prediction_length,
                              input_var=args.input_var,
                              target_var=args.target_var,
                              train_ratio=args.train_ratio,
                              val_ratio=args.val_ratio,
                              batch_size=args.batch_size)

    train_loader, val_loader, test_loader = processor.get_dataloaders()

    # Initialize model
    print("==> Initialize model ...")
    model = SimpleGRU(input_dim=len(args.input_var), n_steps=args.prediction_length, n_units=args.hidden_units)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)
    loss = nn.MSELoss()
    min_val_loss = 9999
    counter = 0

    # Train
    print("==> Start training ...")
    model.train()

    for i in range(args.epochs):
        mse_train = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            opt.zero_grad()
            y_pred = model(batch_x)
            l = loss(y_pred, batch_y)
            l.backward()
            mse_train += l.item() * batch_x.shape[0]
            opt.step()
        epoch_scheduler.step()

        with torch.no_grad():
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                mse_val += loss(output, batch_y).item() * batch_x.shape[0]
        preds = np.concatenate(preds, axis=0)  # 确保多步预测结果正确拼接
        true = np.concatenate(true, axis=0)

        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            print("Saving...")
            torch.save(model.state_dict(), os.path.join(save_dir, 'exp.pt'))
            counter = 0
        else:
            counter += 1

        if counter == args.patience:
            print("counter == patience, early stopping")
            break

        print("Iter: ", i, "train: ", (mse_train / processor.train_size) ** 0.5, "val: ",
              (mse_val / processor.val_size) ** 0.5)

        if (i % 10 == 0):
            preds = processor.inverse_transform_y(preds)
            true = processor.inverse_transform_y(true)
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            print("lr: ", opt.param_groups[0]["lr"])
            print(f"mse: {mse:.4f}, mae: {mae:.4f}")

            # plt.figure(figsize=(20, 10))
            # for j in range(n_steps):
            #     plt.plot(preds[:, j], label=f'Predicted step {j+1}')
            #     plt.plot(true[:, j], label=f'True step {j+1}')
            # plt.legend()
            # plt.show()

    # Prediction
    model.load_state_dict(torch.load(os.path.join(save_dir, 'exp.pt')))
    model.to(device)
    model.eval()

    with torch.no_grad():
        mse_val = 0
        preds = []
        true = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)

            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss(output, batch_y).item() * batch_x.shape[0]

    # 变成 NumPy 数组
    preds = np.concatenate(preds, axis=0)  # 形状 (样本数, n_steps)
    true = np.concatenate(true, axis=0)  # 形状 (样本数, n_steps)
    # 逆归一化
    preds = processor.inverse_transform_y(preds)
    true = processor.inverse_transform_y(true)

    # 评估指标
    evaluator = PredictionEvaluator(prediction_length=args.prediction_length, save_path=save_dir)
    metrics = evaluator.compute_metrics(true, preds)
    evaluator.plot_predictions(true, preds)
    evaluator.plot_r2_scores(metrics['R2'])
    evaluator.count_model_parameters(model)


if __name__ == '__main__':
    main()
