import matplotlib.pyplot as plt
import csv


epochs = []
recalls = []
ndcgs = []

with open('logs/training_metrics_shifted.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epoch = int(row['epoch'])
        if (epoch - 55) % 10 == 0:
            epochs.append(epoch)
            recalls.append(float(row['recall']))
            ndcgs.append(float(row['ndcg']))

plt.figure(figsize=(12, 8))
plt.plot(epochs, recalls, label="Recall@20")
# plt.plot(data["epoch"], data["precision"], label="Precision@20")
plt.plot(epochs, ndcgs, label="NDCG@20")
plt.title('Recall and NDCG over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)

# Lưu ảnh ra file
plt.savefig('recall_ndcg_plot.png', dpi=300)

plt.show()


# import pandas as pd

# # Đọc file CSV hiện tại
# df = pd.read_csv('logs/training_metrics.csv')

# # Tịnh tiến epoch lại
# # Giả sử bắt đầu từ epoch gốc = 55, bước nhảy là 5
# start_epoch = 55
# step = 5

# df['epoch'] = df['epoch'].apply(lambda x: start_epoch + x * step)

# # Ghi lại file mới (hoặc ghi đè file cũ)
# df.to_csv('logs/training_metrics_shifted.csv', index=False)
