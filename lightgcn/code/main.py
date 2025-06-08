import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import json

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metric, model, path):
        score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)


Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

loss_list = []
epoch_list = []
recall_list = []
precision_list = []
ndcg_list = []
early_stopping = EarlyStopping(patience=10, verbose=True)

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch > 0 and epoch %10 == 0:
            cprint("[TEST]")
            recall, precision, ndcg = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            # Early stopping checks based on Recall@K
            early_stopping(recall,  Recmodel, weight_file)

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            recall_list.append(recall)
            precision_list.append(precision)
            ndcg_list.append(ndcg)
            loss_list.append(loss)
            epoch_list.append(epoch)
            results = {
                "epoch": epoch_list,
                "loss": loss_list,
                "recall": recall_list,
                "precision": precision_list,
                "ndcg": ndcg_list
            }
            with open("training_metrics.json", "w") as f:
                json.dump(results, f, indent=4)
            
        loss, output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        
finally:
    if world.tensorboard:
        w.close()