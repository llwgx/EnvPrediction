import pandas as pd
import numpy as np
import torch
import os
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class DataProcessor:
    def __init__(self, file_path, windows, pred_len, input_var, target_var,
                 train_ratio, val_ratio, batch_size,norm_method='min-max', debug=False):
        self.file_path = file_path
        self.windows = windows
        self.pred_len = pred_len
        self.input_var = input_var
        self.target_var = target_var
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.debug = debug
        self.norm_method = norm_method  # 新增参数

        # 变量初始化
        self.train_size = None
        self.val_size = None
        self.y_train_min = None
        self.y_train_max = None
        self.y_train_mean = None
        self.y_train_std = None

    def read_data(self):
        df = pd.read_csv(self.file_path, index_col=0, parse_dates=True, nrows=250 if self.debug else None)

        # 提取输入特征
        feature_data = df[self.input_var]
        feature_cols = list(feature_data.columns)

        # 构造输入 x
        x = np.zeros((len(df), self.windows, len(feature_cols)))
        for i, name in enumerate(feature_cols):
            for j in range(self.windows):
                x[:, j, i] = df[name].shift(self.windows - j - 1).fillna(method='bfill')

        # 构造目标 y
        y = np.zeros((len(df), self.pred_len))
        for i in range(self.pred_len):
            y[:, i] = df[self.target_var].shift(-i - 1).fillna(method='ffill')

        return x, y

    def normalize_data(self, x, y):
        """ 归一化数据，确保仅使用训练集的统计信息 """
        train_bound = int(self.train_ratio * len(x))
        val_bound = int((self.train_ratio + self.val_ratio) * len(x))

        # 划分数据集
        x_train, x_val, x_test = x[:train_bound], x[train_bound:val_bound], x[val_bound:]
        y_train, y_val, y_test = y[:train_bound], y[train_bound:val_bound], y[val_bound:]

        if self.norm_method == 'min-max':
            # 计算训练集的最小值和最大值（避免数据泄露）
            x_train_min, x_train_max = x_train.min(axis=0), x_train.max(axis=0)
            self.y_train_min, self.y_train_max = y_train.min(axis=0), y_train.max(axis=0)  # 存储 y 的归一化参数

            # 归一化
            x_train = (x_train - x_train_min) / (x_train_max - x_train_min + 1e-9)
            x_val = (x_val - x_train_min) / (x_train_max - x_train_min + 1e-9)
            x_test = (x_test - x_train_min) / (x_train_max - x_train_min + 1e-9)
            y_train = (y_train - self.y_train_min) / (self.y_train_max - self.y_train_min + 1e-9)
            y_val = (y_val - self.y_train_min) / (self.y_train_max - self.y_train_min + 1e-9)
            y_test = (y_test - self.y_train_min) / (self.y_train_max - self.y_train_min + 1e-9)
        elif self.norm_method == 'z-score':
            x_train_mean, x_train_std = x_train.mean(axis=0), x_train.std(axis=0)
            self.y_train_mean = y_train.mean(axis=0)
            self.y_train_std = y_train.std(axis=0)

            x_train = (x_train - x_train_mean) / (x_train_std + 1e-9)
            x_val = (x_val - x_train_mean) / (x_train_std + 1e-9)
            x_test = (x_test - x_train_mean) / (x_train_std + 1e-9)
            y_train = (y_train - self.y_train_mean) / (self.y_train_std + 1e-9)
            y_val = (y_val - self.y_train_mean) / (self.y_train_std + 1e-9)
            y_test = (y_test - self.y_train_mean) / (self.y_train_std + 1e-9)

        else:
            raise ValueError("Unsupported normalization method. Choose 'min-max' or 'z-score'.")

        return x_train, x_val, x_test, y_train, y_val, y_test

    def get_dataloaders(self):
        """ 读取数据、归一化并返回 PyTorch DataLoader """
        x, y = self.read_data()
        x_train, x_val, x_test, y_train, y_val, y_test = self.normalize_data(x, y)

        # 转换为 Tensor
        x_train_t = torch.Tensor(x_train)
        x_val_t = torch.Tensor(x_val)
        x_test_t = torch.Tensor(x_test)
        y_train_t = torch.Tensor(y_train)
        y_val_t = torch.Tensor(y_val)
        y_test_t = torch.Tensor(y_test)

        # 记录训练集和验证集的大小
        self.train_size = len(x_train_t)
        self.val_size = len(x_val_t)

        # 创建 DataLoader
        train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def inverse_transform_y(self, y_normalized):
        """ 反归一化目标变量 y """
        if self.norm_method == 'min-max':
            return y_normalized * (self.y_train_max - self.y_train_min + 1e-9) + self.y_train_min
        elif self.norm_method == 'z-score':
            return y_normalized * (self.y_train_std + 1e-9) + self.y_train_mean
        else:
            raise ValueError("Unsupported normalization method for inverse transform.")

    def save_stats_txt(self, save_path):
        """保存归一化相关信息为纯文本格式"""
        lines = []

        lines.append(f"Normalization Method: {self.norm_method}")
        lines.append(f"Train Size: {self.train_size}")
        lines.append(f"Validation Size: {self.val_size}")
        lines.append("")

        if self.y_train_min is not None:
            lines.append(f"y_train_min: {np.array2string(self.y_train_min, separator=', ')}")
        if self.y_train_max is not None:
            lines.append(f"y_train_max: {np.array2string(self.y_train_max, separator=', ')}")
        if self.y_train_mean is not None:
            lines.append(f"y_train_mean: {np.array2string(self.y_train_mean, separator=', ')}")
        if self.y_train_std is not None:
            lines.append(f"y_train_std: {np.array2string(self.y_train_std, separator=', ')}")

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Human-readable stats saved to: {save_path}")


class PredictionEvaluator:
    def __init__(self, prediction_length, save_path):
        self.prediction_length = prediction_length
        self.save_path = save_path

    def compute_metrics(self, true, preds):
        """计算 MSE、MAE、RMSE 和 R²，并保存到 CSV"""
        mse = mean_squared_error(true, preds, multioutput='raw_values')
        mae = mean_absolute_error(true, preds, multioutput='raw_values')
        rmse = np.sqrt(mse)
        r2_scores = [r2_score(true[:, j], preds[:, j]) for j in range(self.prediction_length)]

        metrics_df = pd.DataFrame({
            'Step': range(1, self.prediction_length + 1),
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2_scores
        })
        save_file = os.path.join(self.save_path, "metrics.csv")
        metrics_df.to_csv(save_file, index=False)
        print(f"Metrics saved to {save_file}")

        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2_scores
        }

    def plot_predictions(self, true, preds):
        """绘制预测结果图"""
        plt.figure(figsize=(20, 10))

        for j in range(self.prediction_length):
            plt.plot(preds[:, j], label=f'Predicted step {j + 1}', linestyle='--')

        plt.plot(true[:, 0], label='True step 1', color='black', linewidth=2)

        true_max = np.max(true, axis=1)
        true_min = np.min(true, axis=1)
        plt.fill_between(range(len(true)), true_min, true_max, color='gray', alpha=0.2, label="True range")

        plt.xlabel("样本")
        plt.ylabel("温度")
        plt.legend()
        plt.title(f"LSTM {self.prediction_length} 步预测结果")

        # 保存图片
        plot_path = os.path.join(self.save_path, "predictions.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Prediction plot saved to {plot_path}")

        plt.close()  # 关闭图像，避免显示

        # 保存预测数据
        true_path = os.path.join(self.save_path, "true.npy")
        preds_path = os.path.join(self.save_path, "preds.npy")
        np.save(true_path, true)
        np.save(preds_path, preds)

    def plot_r2_scores(self, r2_scores):
        """绘制 R² 变化图"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.prediction_length + 1), r2_scores, marker='o', linestyle='-')
        plt.xlabel("预测时间步")
        plt.ylabel("R2 分数")
        plt.title("多步预测的 R2 变化")
        plt.ylim(0, 1)
        plt.grid(True)

        # 保存图片
        r2_plot_path = os.path.join(self.save_path, "r2_scores.png")
        plt.savefig(r2_plot_path, bbox_inches='tight')
        print(f"r2 score plot saved to {r2_plot_path}")

        plt.close()

    def count_model_parameters(self, model):
        """计算模型参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        # 保存参数信息到文本文件
        param_file = os.path.join(self.save_path, "model_parameters.txt")
        with open(param_file, "w") as f:
            f.write(f"Total number of parameters: {total_params}\n")

        print(f"Model parameters saved to {param_file}")

        return total_params