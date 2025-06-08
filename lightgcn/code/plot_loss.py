# visualize_results.py
import json
import matplotlib.pyplot as plt

with open("logs/training_metrics.json", "r") as f:
    data = json.load(f)

plt.figure(figsize=(12, 8))
# plt.plot(data["epoch"], data["loss"], label="Loss")
plt.plot(data["epoch"], data["recall"], label="Recall@20")
# plt.plot(data["epoch"], data["precision"], label="Precision@20")
plt.plot(data["epoch"], data["ndcg"], label="NDCG@20")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()
plt.title("Training Metrics Over Epochs")
plt.grid()
plt.savefig("training_metrics.png")
plt.show()
