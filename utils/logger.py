# logger.py
import csv
import json
import os

class TrainingLogger:
    def __init__(self, folder="logs"):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.csv_path = os.path.join(folder, "training_log.csv")
        self.json_path = os.path.join(folder, "training_log.json")

        # Initialize files if not already
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Loss"])

        if not os.path.exists(self.json_path):
            with open(self.json_path, "w") as f:
                json.dump([], f)

    def log(self, epoch, loss):
        """
        Logs losses for each epoch.
        Accepts either:
        - a single numeric loss, or
        - a dictionary of multiple losses (e.g., {"D_Loss": x, "G_Loss": y})
        """

        # Handle both scalar and dictionary losses
        if isinstance(loss, dict):
            log_entry = {"Epoch": epoch, **loss}
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss.items()])
        else:
            log_entry = {"Epoch": epoch, "Loss": loss}
            loss_str = f"Loss: {loss:.4f}"

        # Print to terminal
        print(f"Logged: Epoch {epoch} | {loss_str}")

        # Update CSV
        csv_fields = log_entry.keys()
        write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(log_entry)

        # Update JSON
        if os.path.exists(self.json_path):
            with open(self.json_path, "r+") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=4)
        else:
            with open(self.json_path, "w") as f:
                json.dump([log_entry], f, indent=4)
