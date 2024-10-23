import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# 定义注意力层
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x的形状[batch, seq_len, features]
        weights = self.attention(x)  # [batch, seq_len, 1]
        weighted = torch.mul(x, weights.expand_as(x))  # [batch, seq_len, features]
        return weighted.sum(1)  # [batch, features]


# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, bidirectional, dropout):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_layer_size, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        self.attention = Attention(hidden_layer_size * (2 if bidirectional else 1))
        self.linear = nn.Linear(hidden_layer_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, num_directions * hidden_size]
        attention_out = self.attention(lstm_out)  # [batch, num_directions * hidden_size]
        predictions = self.linear(attention_out)  # [batch, output_size]
        return predictions


# 创建输入序列和输出标签
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# 训练模型函数
def train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, epochs):
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, labels in val_loader:
                y_pred = model(seq)
                val_loss += loss_function(y_pred, labels).item()

        training_losses.append(total_loss / len(train_loader))
        validation_losses.append(val_loss / len(val_loader))
        scheduler.step(val_loss)
        print(f'第 {epoch} 轮, 训练损失: {training_losses[-1]:.4f}, 验证损失: {validation_losses[-1]:.4f}')

    print('训练完成。')
    return training_losses, validation_losses


def predict_future(model, initial_seq, num_days, scaler):
    model.eval()
    input_tensor = initial_seq
    predictions = []

    with torch.no_grad():
        for _ in range(num_days):
            pred = model(input_tensor)
            # 归一化预测值，并更新输入序列
            pred_value = pred.item()
            predictions.append(pred_value)
            # 更新输入序列
            new_data_point = torch.tensor([[pred_value]], dtype=torch.float32).unsqueeze(0)
            input_tensor = torch.cat((input_tensor[:, 1:, :], new_data_point), dim=1)

    # 反归一化预测值
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    predictions_rounded = np.rint(predictions).astype(int)
    return predictions_rounded

file_path = 'comment_count.csv'
data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
model = ARIMA(data['num'], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=5)
forecast_rounded = forecast.round().astype(int)
forecast_list = forecast_rounded.tolist()

# 加载和预处理数据
df = pd.read_csv('comment_count.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
scaler = MinMaxScaler(feature_range=(-1, 1))
df['num_normalized'] = scaler.fit_transform(df['num'].values.reshape(-1, 1))

# 参数
seq_length = 4
batch_size = 1
input_size = 1
hidden_layer_size = 50
output_size = 1
learning_rate = 0.03
epochs = 100
num_layers = 2
bidirectional = True
k_folds = 5  # k折交叉验证

# 准备数据
data = df['num_normalized'].values

# k折交叉验证
kf = KFold(n_splits=k_folds, shuffle=True)

all_training_losses = []
all_validation_losses = []

for fold, (train_index, val_index) in enumerate(kf.split(data)):
    print(f'正在进行第 {fold + 1} 折...')

    train_data = data[train_index]
    val_data = data[val_index]

    train_sequences = create_inout_sequences(train_data, seq_length)
    val_sequences = create_inout_sequences(val_data, seq_length)

    train_seqs, train_labels = zip(*train_sequences)
    val_seqs, val_labels = zip(*val_sequences)

    train_seqs_tensor = torch.FloatTensor(train_seqs).view(-1, seq_length, 1)
    train_labels_tensor = torch.FloatTensor(train_labels).view(-1, 1)
    val_seqs_tensor = torch.FloatTensor(val_seqs).view(-1, seq_length, 1)
    val_labels_tensor = torch.FloatTensor(val_labels).view(-1, 1)

    train_dataset = TensorDataset(train_seqs_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_seqs_tensor, val_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = LSTM(input_size, hidden_layer_size, output_size, num_layers, bidirectional, dropout=0.2)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    training_losses, validation_losses = train_model(model, train_loader, val_loader, loss_function, optimizer,
                                                     scheduler, epochs)
    all_training_losses.append(training_losses)
    all_validation_losses.append(validation_losses)

# 绘制损失趋势图
mean_training_losses = np.mean(all_training_losses, axis=0)
mean_validation_losses = np.mean(all_validation_losses, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(mean_training_losses, label='Average training loss')
plt.plot(mean_validation_losses, label='Average validation loss')
plt.title('Training and validation loss')
plt.xlabel('round')
plt.ylabel('loss')
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')
print('模型已保存。')

# 预测未来的天数
last_seq = torch.FloatTensor(df['num_normalized'].values[-seq_length:]).view(1, seq_length, 1)
num_days = 5  # 想要预测的天数
future_predictions = predict_future(model, last_seq, num_days, scaler)

print(f"未来整数预测: {forecast_list}")

