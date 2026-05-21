import os
import pandas as pd
import matplotlib.pyplot as plt


EXPERIMENT_DIR = r"checkpoints/bilstm_seq15_vocab10000_20260513_133233"

history_path = os.path.join(EXPERIMENT_DIR, "training_history.csv")
history = pd.read_csv(history_path)

print(history.columns)

# Plot accuracy and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history["accuracy"], label="Training accuracy")
plt.plot(history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXPERIMENT_DIR, "accuracy_curve.png"), dpi=300)
plt.show()

# Plot top-5 accuracy and validation top-5 accuracy
plt.figure(figsize=(8, 5))
plt.plot(history["top_5_accuracy"], label="Training top-5 accuracy")
plt.plot(history["val_top_5_accuracy"], label="Validation top-5 accuracy")
plt.xlabel("Epoch")
plt.ylabel("Top-5 Accuracy")
plt.title("Training and Validation Top-5 Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXPERIMENT_DIR, "top5_accuracy_curve.png"), dpi=300)
plt.show()

# Plot loss and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history["loss"], label="Training loss")
plt.plot(history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXPERIMENT_DIR, "loss_curve.png"), dpi=300)
plt.show()