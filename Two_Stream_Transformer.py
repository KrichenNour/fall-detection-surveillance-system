import os
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from glob import glob
import numpy as np
from tqdm import tqdm
import pandas as pd

import json
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, classification_report # type: ignore
import matplotlib.pyplot as plt

def horizental_flip(fg, flow):
    """Horizontally flip fg and flow data."""
    fg_flipped = np.flip(fg, axis=-1).copy()  # flip width axis
    flow_flipped = np.flip(flow, axis=-1).copy()  # flip width axis
    flow_flipped[:,0,:,:] *= -1  # invert x-direction flow
    return fg_flipped, flow_flipped

def temporal_jitter(fg, flow, max_jitter=5):
    """
    Temporal jitter by shifting sequence and padding.
    fg:   (T, 1, H, W)
    flow: (T, 2, H, W)
    """
    jitter = random.randint(-max_jitter, max_jitter)

    if jitter > 0:
        fg = np.concatenate([fg[jitter:], fg[-1:].repeat(jitter, axis=0)], axis=0)
        flow = np.concatenate([flow[jitter:], flow[-1:].repeat(jitter, axis=0)], axis=0)

    elif jitter < 0:
        fg = np.concatenate([fg[:1].repeat(-jitter, axis=0), fg[:jitter]], axis=0)
        flow = np.concatenate([flow[:1].repeat(-jitter, axis=0), flow[:jitter]], axis=0)

    return fg, flow

def flow_noise(flow, sigma=0.02):
    noise = np.random.normal(0,sigma,flow.shape).astype(np.float32)
    return np.clip(flow + noise, -1.0, 1.0)


class FGFLOWDataset(Dataset):
    """
    Dataset for loading foreground masks and optical flow data.
    foreground_masks: List of file paths to foreground mask .npy files / shape (63,224,224)
    optical flow data: List of file paths to optical flow .npy files / shape (63,224,224,2)
    labels: List of integer labels (0 or 1)
    """
    def __init__(self,fg_fall_dir,fg_no_fall_dir, flow_fall_dir,flow_no_fall_dir,labels:dict,flow_clip:float=20.0,transform=None):
        super().__init__()
        self.fg_fall_dir = fg_fall_dir
        self.fg_no_fall_dir = fg_no_fall_dir
        self.flow_fall_dir = flow_fall_dir
        self.flow_no_fall_dir = flow_no_fall_dir

        self.labels = labels
        self.flow_clip = flow_clip
        self.transform = transform

        # Only use stems that are in the labels dict
        self.stems = list(labels.keys())

        self.transform = transform
        """
        self.stems = []
        self.transform_flags = []
        for stem in labels.keys():
            self.stems.append(stem)
            self.transform_flags.append(False)  # original
            if self.transform and random.random() < 0.8:  # 80% chance to add augmented version
                self.stems.append(stem)
                self.transform_flags.append(True)   # augmented   
        """
        print(f"Dataset initialized with {len(self.stems)} samples")
        
    def __len__(self):
        return len(self.stems)
        
    def __getitem__(self,idx:int):
        stem = self.stems[idx]
        
        # Check if this stem exists in labels
        if stem not in self.labels:
            raise KeyError(f"Stem {stem} not found in labels dict")
        
        label = self.labels[stem]
        
        # Determine which directory to look in based on label
        if label == 1:  # Fall
            fg_path = os.path.join(self.fg_fall_dir, stem + "_fg.npy")
            flow_path = os.path.join(self.flow_fall_dir, stem + "_flow.npy")
        else:  # No Fall
            fg_path = os.path.join(self.fg_no_fall_dir, stem + "_fg.npy")
            flow_path = os.path.join(self.flow_no_fall_dir, stem + "_flow.npy")
        
        # Check if files exist
        if not os.path.exists(fg_path):
            raise FileNotFoundError(f"FG file not found: {fg_path}")
        if not os.path.exists(flow_path):
            raise FileNotFoundError(f"Flow file not found: {flow_path}")
        
        fg = np.load(fg_path, allow_pickle=False)  # shape (seq_len,224,224)
        flow = np.load(flow_path, allow_pickle=False)  # shape (seq_len,224,224,2)

        fg_frames = [f for f in fg]
        flow_frames = [f for f in flow]

        ## standardize foreground masks to 0/1
        fg_stack = np.stack(fg_frames,axis=0).astype(np.float32)  # shape (seq_len,224,224)
        if fg_stack.max() > 1:
            fg_stack = fg_stack / 255.0
        fg_stack = np.expand_dims(fg_stack, axis=1)  # shape (seq_len,1,224,224)

        ## flow clip to [-1,1]
        flow_stack = np.stack(flow_frames,axis=0).astype(np.float32)  # shape (seq_len,224,224,2)
        flow_stack = np.transpose(flow_stack, (0,3,1,2))  # shape (seq_len,2,224,224) (B,C,H,W)
        flow_stack = np.clip(flow_stack, -self.flow_clip, self.flow_clip) / self.flow_clip  # normalize to [-1,1]
        
        if aug and self.transform:
            fg_flipped, flow_flipped   = horizental_flip(fg_stack, flow_stack)
            fg_jittered, flow_jittered = temporal_jitter(fg_flipped, flow_flipped, max_jitter=5)
        fg_stack = fg_stack.append(fg_jittered) 
        flow_stack = flow_stack.append(flow_jittered) 
        if self.transform:
            if random.random() < 0.5:
                fg_stack, flow_stack = horizental_flip(fg_stack, flow_stack)

            if random.random() < 0.5:
                fg_stack, flow_stack = temporal_jitter(fg_stack, flow_stack, max_jitter=5)

            if random.random() < 0.2:
                flow_stack = flow_noise(flow_stack)
        
        fg_tensors = torch.from_numpy(fg_stack)  # shape (seq_len,1,224,224)
        flow_tensors = torch.from_numpy(flow_stack)  # shape (seq_len,2,224,224)
        
        return fg_tensors, flow_tensors, label

class PatchEmbedding(nn.Module):
    def __init__(self,in_ch:int,patch_size:int,d_model:int,img_size:int):
        super().__init__()
        self.patch_size = patch_size       
        self.proj = nn.Conv2d(in_ch,d_model,kernel_size=patch_size,stride=patch_size)
    def forward(self,x): 
        x= self.proj(x) # (B*T,d_model,H/patch_size,W/patch_size)
        B,D,h,w =x.shape
        x = x.flatten(2).transpose(1,2)  # (B,num_patches,d_model)
        return x 


def transformer_encdoer_layer(d_model, nhead, mlp_ratio, depth,dropout=0.3):
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=int(d_model * mlp_ratio), dropout=dropout,
                                                activation='gelu'
                                                )
    return nn.TransformerEncoder(encoder_layer, num_layers=depth)

class TwoStreamTransformer(nn.Module):
    def __init__(self,img_size=224,patch_size=64,fg_in_ch=1,flow_in_ch=2,d_model=64,depth=1,num_heads=4,mlp_ratio=2,dropout=0.3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.d_model = d_model
        self.seq_tokens = 63*self.num_patches_per_frame   # number of tokens per video stream

        self.fg_patch_embed   = PatchEmbedding(fg_in_ch,patch_size,d_model,img_size)
        self.flow_patch_embed = PatchEmbedding(flow_in_ch,patch_size,d_model,img_size)

        self.cls_fg_token   = nn.Parameter(torch.zeros(1,1,d_model))
        self.cls_flow_token = nn.Parameter(torch.zeros(1,1,d_model))    

        self.pos_embed_fg   = nn.Parameter(torch.zeros(1,1+self.seq_tokens,d_model))
        self.pos_embed_flow = nn.Parameter(torch.zeros(1,1+self.seq_tokens,d_model))

        self.encoder_fg   = transformer_encdoer_layer(d_model,num_heads,mlp_ratio,depth,dropout)
        self.encoder_flow = transformer_encdoer_layer(d_model,num_heads,mlp_ratio,depth,dropout)

        # projection head for cls tokens : takes the two cls tokens concatenated from both streams
        self.head_dim= d_model*2
        self.classifier = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.head_dim // 2, 1)
        )

        # init 
        nn.init.trunc_normal_(self.pos_embed_fg, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_flow, std=0.02)
        nn.init.trunc_normal_(self.cls_fg_token, std=0.02)
        nn.init.trunc_normal_(self.cls_flow_token, std=0.02)
    
    
    def forward(self,fg_x,flow_x):
        B,T,C1,H,W = fg_x.shape    # fg_x: (B,63,1,224,224)
        B,T,C2,H,W = flow_x.shape  # flow_x: (B,63,2,224,224)
    
        fg_x = fg_x.view(B*T,C1,H,W)
        flow_x = flow_x.view(B*T,C2,H,W)

        # patch embdedding
        fg_x = self.fg_patch_embed(fg_x)         # (B*T,num_patches_per_frame,d_model)
        flow_x = self.flow_patch_embed(flow_x)   # (B*T,num_patches_per_frame,d_model)
    
        
        P = fg_x.size(1)  # num_patches_per_frame
        fg_tokens = fg_x.reshape(B,T*P,self.d_model)      # (B,total_number_of_patches,d_model)
        flow_tokens = flow_x.reshape(B,T*P,self.d_model)  # (B,total_number_of_patches,d_model)

        cls_fg = self.cls_fg_token.expand(B,-1,-1)         # (B,1,d_model)
        cls_flow = self.cls_flow_token.expand(B,-1,-1)     # (B,1,d_model)
        fg_tokens = torch.cat((cls_fg,fg_tokens),dim=1)         # (B,1+seq_len*num_patches_per_frame,d_model)
        flow_tokens = torch.cat((cls_flow,flow_tokens),dim=1)   # (B,1+seq_len*num_patches_per_frame,d_model)

        ## add pos embedding
        fg_tokens = fg_tokens + self.pos_embed_fg
        flow_tokens = flow_tokens + self.pos_embed_flow

        fg_out   = self.encoder_fg(fg_tokens.transpose(0, 1)).transpose(0, 1)         # (B,1+seq_len*num_patches_per_frame,d_model)
        flow_out = self.encoder_flow(flow_tokens.transpose(0, 1)).transpose(0, 1)   # (B,1+seq_len*num_patches_per_frame,d_model)

        cls_fg_out   = fg_out[:,0,:]    # (B,d_model)
        cls_flow_out = flow_out[:,0,:]  # (B,d_model)
        fused = torch.cat((cls_fg_out,cls_flow_out),dim=1)  # (B,2*d_model)

        logits = self.classifier(fused)  # (B,1)


        return logits.squeeze(1)  # (B,)


def collate_fn(batch):
    fg_tensors = torch.stack([item[0] for item in batch], dim=0)      # (B,seq_len,1,224,224)
    flow_tensors = torch.stack([item[1] for item in batch], dim=0)    # (B,seq_len,2,224,224)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)  # (B,)
    return fg_tensors, flow_tensors, labels

def train_one_epoch(model,dataloader,optimizer,device,criterion):
    model.train()
    total_loss = 0.0
    correct =0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader,desc="Training")
    for fg_batch, flow_batch, labels in pbar:
        optimizer.zero_grad()

        fg_batch   = fg_batch.to(device)
        flow_batch = flow_batch.to(device)
        labels     = labels.to(device)
        
        
        outputs = model(fg_batch, flow_batch)
        loss    = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item() * fg_batch.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    

        pbar.set_postfix({'loss': total_loss / total, 'acc': correct/total})
    avg_loss = total_loss / total
    avg_acc = correct / total
    all_preds = np.array(all_preds)
    all_probs =  np.array(all_probs)        
    all_labels = np.array(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    return avg_loss, avg_acc, precision, recall, f1,all_probs, all_labels

def validate(model,dataloader,device,criterion):
    model.eval()
    total_loss = 0.0
    correct =0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader,desc="Validation")
        for fg_batch, flow_batch, labels in pbar:
            fg_batch = fg_batch.to(device)
            flow_batch = flow_batch.to(device)
            labels = labels.to(device)

            outputs = model(fg_batch, flow_batch)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * fg_batch.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'val_loss': total_loss / total, 'val_acc': correct/total})
    all_preds = np.array(all_preds)
    all_probs =  np.array(all_probs)
    all_labels = np.array(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    avg_loss = total_loss / total
    avg_acc = correct / total
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, avg_acc, precision, recall, f1,all_probs, all_labels,cm
def plot_metrics(history, save_dir="./training_aug_2_results"):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.close()
    
    # Plot Precision, Recall, F1
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_precision'], 'b-', label='Train Precision')
    plt.plot(epochs, history['val_precision'], 'r-', label='Val Precision')
    plt.plot(epochs, history['train_recall'], 'g-', label='Train Recall')
    plt.plot(epochs, history['val_recall'], 'm-', label='Val Recall')
    plt.plot(epochs, history['train_f1'], 'c-', label='Train F1')
    plt.plot(epochs, history['val_f1'], 'y-', label='Val F1')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    plt.close()
    
    # Plot ROC Curve (using last epoch data)
    fpr, tpr, _ = roc_curve(history['val_labels'][-1], history['val_probs'][-1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    # Plot Confusion Matrix (last epoch)
    cm = history['val_confusion_matrix'][-1]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Last Epoch)')
    plt.colorbar()
    classes = ['No Fall', 'Fall']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"\n All plots saved to {save_dir}")

def save_training_history(history, save_dir="./training_aug_2_results"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as JSON (excluding numpy arrays)
    metrics_dict = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'train_precision': history['train_precision'],
        'val_precision': history['val_precision'],
        'train_recall': history['train_recall'],
        'val_recall': history['val_recall'],
        'train_f1': history['train_f1'],
        'val_f1': history['val_f1'],
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    # Save as CSV for easy analysis
    df = pd.DataFrame(metrics_dict)
    df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    
    # Save final classification report
    y_true = history['val_labels'][-1]
    y_pred = (np.array(history['val_probs'][-1]) >= 0.5).astype(int)
    
    report = classification_report(y_true, y_pred, target_names=['No Fall', 'Fall'])
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write("Final Classification Report (Last Epoch)\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    
    print(f" Training history saved to {save_dir}")
    print("\nFinal Classification Report:")
    print(report)


if __name__ == "__main__":
    fg_fall_dir = "./Processed_For_DL/Fall/fg"
    flow_fall_dir = "./Processed_For_DL/Fall/flow"
    fg_no_fall_dir = "./Processed_For_DL/No_Fall/fg"
    flow_no_fall_dir = "./Processed_For_DL/No_Fall/flow"
    # create labels
    labels_csv = "./Processed_For_DL/labels.csv"
    batch_size = 16
    epochs = 30
    lr = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    labels = {}
    labels_df = pd.read_csv(labels_csv)
    for _, row in labels_df.iterrows():
        stem = row['filename']
        label = row['label']
        labels[stem] = label

    
    # split stems into train/val
    all_stems= sorted(list(labels.keys()))
    print("###"*30)
    print(f"Total samples: {len(all_stems)}")
    print(f"Sample stems: {all_stems[:10]}")
    print("###"*30)

    random.shuffle(all_stems)
    split = int(0.8 * len(all_stems))
    train_stems = set(all_stems[:split])
    val_stems = set(all_stems[split:])
    
    def filter_by_stems(fg_fall_dir,fg_no_fall_dir, flow_fall_dir,flow_no_fall_dir, stem_set):
        """Returns a labels dict filtered for a specific directory and stem subset."""
        filtered = {}
        for stem in stem_set:
            fg_path_fall = os.path.join(fg_fall_dir, stem + "_fg.npy")
            fg_path_no_fall = os.path.join(fg_no_fall_dir, stem + "_fg.npy")
            flow_path_fall = os.path.join(flow_fall_dir, stem + "_flow.npy")
            flow_path_no_fall = os.path.join(flow_no_fall_dir, stem + "_flow.npy")
            if (os.path.exists(fg_path_fall) or os.path.exists(fg_path_no_fall)) and (os.path.exists(flow_path_fall) or os.path.exists(flow_path_no_fall)):
                filtered[stem] = labels[stem]
            else: 
                print(f"Warning: Missing files for stem {stem}, skipping.")
        return filtered

    # create datasets
    train_labels    = filter_by_stems(fg_fall_dir,fg_no_fall_dir, flow_fall_dir,flow_no_fall_dir, train_stems)
    val_labels      = filter_by_stems(fg_fall_dir, fg_no_fall_dir, flow_fall_dir, flow_no_fall_dir, val_stems)


    train_dataset    = FGFLOWDataset(fg_fall_dir,fg_no_fall_dir, flow_fall_dir,flow_no_fall_dir, train_labels,transform=True)
    val_dataset      = FGFLOWDataset(fg_fall_dir, fg_no_fall_dir, flow_fall_dir, flow_no_fall_dir, val_labels,transform=False)     

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=6,pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=6,pin_memory=False)

    model = TwoStreamTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=0.05) 
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    best_val_acc = 0.0

     # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_f1': [],
        'val_f1': [],
        'val_probs': [],
        'val_labels': [],
        'val_confusion_matrix': []
    }
    fall_count = sum(1 for v in train_labels.values() if v == 1)
    no_fall_count = len(train_labels) - fall_count
    
    print(f"\n📊 Class Distribution:")
    print(f"  Fall: {fall_count} ({fall_count/len(train_labels)*100:.1f}%)")
    print(f"  No Fall: {no_fall_count} ({no_fall_count/len(train_labels)*100:.1f}%)")
    
    # Calculate class weights
    if fall_count != no_fall_count:
        pos_weight = torch.tensor([no_fall_count / fall_count]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  Using pos_weight={pos_weight.item():.2f} for BCEWithLogitsLoss")
    else:
        criterion = nn.BCEWithLogitsLoss()
 
    for epoch in range(epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1,_,_ = train_one_epoch(
                            model,train_loader,optimizer,device,criterion=criterion
                        )
        val_loss, val_acc, val_precision, val_recall, val_f1, val_probs_epoch, val_labels_epoch, val_cm = validate(
                            model,val_loader,device,criterion=criterion
                        )   
        if scheduler:
            scheduler.step()
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        if epoch == epochs - 1:
            history['val_probs'].append(val_probs_epoch)
            history['val_labels'].append(val_labels_epoch)
            history['val_confusion_matrix'].append(val_cm)

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_two_stream_transformer_aug_3.pth")
            print(f"Saved best model (Val Acc: {best_val_acc:.4f})")

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    
    # Save history and generate plots
    save_training_history(history, save_dir="./training_aug_3_results")
    plot_metrics(history, save_dir="./training_aug_3_results")
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")