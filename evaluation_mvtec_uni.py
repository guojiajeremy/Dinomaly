import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from functools import partial
import matplotlib.cm as cm
from skimage.segmentation import find_boundaries

from dataset import get_data_transforms, MVTecDataset
from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2

# MODEL
def get_model(encoder_name='dinov2reg_vit_base_14', device='cuda'):

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
        target_layers = [2,3,4,5,6,7,8,9]

    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
        target_layers = [2,3,4,5,6,7,8,9]

    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4,6,8,10,12,14,16,18]

    else:
        raise ValueError("Unsupported encoder")

    fuse_layer_encoder = [[0,1,2,3],[4,5,6,7]]
    fuse_layer_decoder = [[0,1,2,3],[4,5,6,7]]

    encoder = vit_encoder.load(encoder_name)

    bottleneck = nn.ModuleList([
        bMlp(embed_dim, embed_dim*4, embed_dim, drop=0.2)
    ])

    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2
        )
        for _ in range(8)
    ])

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder
    ).to(device)

    return model

# ANOMALY MAP
def compute_anomaly_map(en_feats, de_feats, image_size=448):

    anomaly_maps = []

    for e, d in zip(en_feats, de_feats):

        diff = 1.0 - nn.functional.cosine_similarity(e, d, dim=1)

        anomaly_maps.append(diff)

    amap = torch.mean(torch.stack(anomaly_maps), dim=0)

    amap = torch.nn.functional.interpolate(
        amap.unsqueeze(1),
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=True
    ).squeeze(1)

    amap = amap.clamp_(0,1).cpu().numpy()

    for i in range(amap.shape[0]):
        amap[i] = gaussian_filter(amap[i], sigma=3)

    return amap

# THRESHOLD
def find_youden_threshold(labels, scores):

    if len(set(labels)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    return thresholds[np.argmax(youden)]

# VISUALIZATION
def save_all_visualizations(img_paths, gt_masks, anomaly_maps, scores, vis_dir):

    os.makedirs(vis_dir, exist_ok=True)

    print(f"Saving {len(img_paths)} visualizations → {vis_dir}")

    for idx,(img_path,gt_np,pred_map,score) in enumerate(
        tqdm(zip(img_paths,gt_masks,anomaly_maps,scores),total=len(img_paths))
    ):

        orig = np.array(Image.open(img_path).convert("RGB"))
        H,W = orig.shape[:2]

        pred_map = np.array(
            Image.fromarray(pred_map).resize((W,H),Image.BILINEAR)
        )

        pred_map = (pred_map-pred_map.min())/(pred_map.ptp()+1e-8)

        # safe GT mask
        if gt_np is not None and isinstance(gt_np,np.ndarray) and gt_np.ndim>=2:

            gt_mask = np.array(
                Image.fromarray(gt_np.astype(np.uint8)).resize((W,H),Image.NEAREST)
            )>0

        else:

            gt_mask = np.zeros((H,W),dtype=bool)

        heatmap = (cm.jet(pred_map)[...,:3]*255).astype(np.uint8)

        overlay = orig.astype(float)

        high_score = pred_map>0.4

        overlay[high_score] = overlay[high_score]*0.6 + np.array([255,0,0])*0.4

        if gt_mask.any():

            boundary = find_boundaries(gt_mask,mode="thick")
            overlay[boundary] = [0,255,0]

        overlay = np.clip(overlay,0,255).astype(np.uint8)

        fig,ax = plt.subplots(1,3,figsize=(18,6))

        ax[0].imshow(orig)
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(heatmap)
        ax[1].set_title(f"Anomaly Map\nMax={score:.4f}")
        ax[1].axis("off")

        ax[2].imshow(overlay)
        ax[2].set_title("Overlay (Red=Pred | Green=GT)")
        ax[2].axis("off")

        plt.tight_layout()

        save_name = f"{idx:04d}_{os.path.basename(img_path)}"

        plt.savefig(os.path.join(vis_dir,save_name),dpi=200,bbox_inches="tight")
        plt.close()

# EVALUATION
def evaluate_and_visualize(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:",device)

    save_dir = os.path.join(args.save_dir,args.save_name)

    model_path = r"D:\Watch_Demo\new_pcb\Dinomaly\saved_results\model.pth"

    eval_dir = os.path.join(save_dir,"eval_results")
    vis_dir = os.path.join(eval_dir,"all_visualizations")

    os.makedirs(vis_dir,exist_ok=True)

    model = get_model(device=device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    image_size=448
    crop_size=392
    data_transform,gt_transform=get_data_transforms(image_size,crop_size)

    scores=[]
    labels=[]
    img_paths=[]
    gt_masks=[]
    anomaly_maps=[]

    for category in args.item_list:
        print("Evaluating",category)

        dataset=MVTecDataset(
            root=os.path.join(args.data_path,category),
            transform=data_transform,
            gt_transform=gt_transform,
            phase="test"
        )

        loader=torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=False)

        with torch.no_grad():

            for img,label,mask,path in tqdm(loader):

                img=img.to(device)

                en,de=model(img)

                amap=compute_anomaly_map(en,de,image_size)

                for i in range(img.shape[0]):
                    
                    p = path[i]

                    score = float(np.max(amap[i]))
                    folder_name = os.path.basename(os.path.dirname(p)).lower()
                    is_anom = 0 if folder_name == "good" else 1
                    scores.append(score)
                    labels.append(is_anom)
                    img_paths.append(p)
                    anomaly_maps.append(amap[i])

                    if mask is not None:
                        gt_masks.append(mask[i].squeeze().cpu().numpy())
                    else:
                        gt_masks.append(None)


    print("\nComputing metrics...")

    auroc = roc_auc_score(labels,scores) if len(set(labels))==2 else 0.5
    print("Image AUROC:",auroc)

    thresh=find_youden_threshold(labels,scores)
    print("Youden threshold:",thresh)

    #SCORE DISTRIBUTION
    good_scores = [s for s,l in zip(scores,labels) if l==0]
    def_scores  = [s for s,l in zip(scores,labels) if l==1]

    max_good_score = max(good_scores) if good_scores else None
    min_def_score  = min(def_scores) if def_scores else None

    # correct midpoint separation
    separation_thresh = None
    if max_good_score is not None and min_def_score is not None:
        separation_thresh = (max_good_score + min_def_score) / 2

    plt.figure(figsize=(10,6))
    plt.hist(
        good_scores,
        bins=50,
        alpha=0.7,
        color="green",
        label="Good Images",
        density=True
    )
    plt.hist(
        def_scores,
        bins=40,
        alpha=0.8,
        color="red",
        label="Defect Images",
        density=True
    )

    # draw separation line
    if separation_thresh is not None:
        plt.axvline(
            separation_thresh,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Separation Threshold = {separation_thresh:.4f}"
        )

    legend_extra=[]
    if max_good_score is not None:
        legend_extra.append(f"Max Good Score = {max_good_score:.4f}")

    if min_def_score is not None:
        legend_extra.append(f"Min Defect Score = {min_def_score:.4f}")

    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.title("Score Distribution")
    plt.grid(alpha=0.3)
    plt.legend(title="\n".join(legend_extra))

    # ensure folder exists
    os.makedirs(eval_dir, exist_ok=True)

    score_plot = os.path.join(eval_dir,"score_distribution.png")
    plt.savefig(score_plot,dpi=300,bbox_inches="tight")
    plt.close()

    print("Score distribution saved:",score_plot)
    print("Max Good Score:", max_good_score)
    print("Min Defect Score:", min_def_score)
    print("Separation Threshold:", separation_thresh)
    
    # VISUALIZATION
    save_all_visualizations(img_paths,gt_masks,anomaly_maps,scores,vis_dir)

    print("Evaluation complete")

# MAIN
if __name__=="__main__":

    import argparse

    parser=argparse.ArgumentParser()

    parser.add_argument('--data_path',type=str,
        default=r'D:\new_pcb\mvtec_anomaly_detection')

    parser.add_argument('--save_dir',type=str,
        default='./saved_results')

    parser.add_argument('--save_name',type=str,
        default='vitill_mvtec_eval')

    parser.add_argument('--item_list',nargs='+',
        default=['back','front'])

    args=parser.parse_args()

    evaluate_and_visualize(args)
