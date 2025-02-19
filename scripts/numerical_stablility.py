# %%
import torch
import sys, os
workdir = os.getcwd()
sys.path.append(os.path.join(workdir,"universal-image-restoration"))
sys.path.append(os.path.join(workdir,"universal-image-restoration/config/daclip-sde"))
import open_clip
from models import create_model
from utils.sde_utils import IRSDE
import options
from data import util
import cv2
from utils.sampler import MRSampler, HeunSampler
from glob import glob
import lpips
from utils.img_utils import tensor2img, calculate_psnr, calculate_ssim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

savedir = os.path.join(workdir,"scripts/numerical_stability")
os.system(f"rm -rf {savedir}/*")
# get config
opt = options.parse(os.path.join(workdir,"universal-image-restoration/config/daclip-sde/options/test.yml"), is_train=False)
opt = options.dict_to_nonedict(opt)
# init model
model = create_model(opt)
device = model.device
clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model = clip_model.to(device)
lpips_fn = lpips.LPIPS(net='vgg').to(device)
# init sde
sde = IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule='cosine', eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
# get test data
srcpath = glob("/mnt/intern/zhejing.la/data/4_daclip/daclip-val/low-light/LQ/*")[0]
tgtpath = glob("/mnt/intern/zhejing.la/data/4_daclip/daclip-val/low-light/GT/*")[0]
srcimg = util.read_img(None, srcpath)
tgtimg = util.read_img(None, tgtpath)
srcimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
tgtimg = cv2.cvtColor(tgtimg, cv2.COLOR_BGR2RGB)
lq4clip = util.clip_transform(srcimg).unsqueeze(0).to(device)
LQ = torch.from_numpy(srcimg).permute(2, 0, 1).float().unsqueeze(0)
GT = torch.from_numpy(tgtimg).permute(2, 0, 1).float().unsqueeze(0)
# get clip features
with torch.no_grad():
    image_context, degra_context = clip_model.encode_image(lq4clip, control=True)
    image_context = image_context.float()
    degra_context = degra_context.float()
noisy_state = sde.noise_state(LQ)
model.feed_data(noisy_state, LQ, GT, text_context=degra_context, image_context=image_context)

def analysis(nfe, parameterization, savedir):
    """
    args:
        nfe: number of sampling steps
        parameterization: parameterization for sampling, choose from ["noise", "data"]
        savedir: directory to save results
    """
    # -----sampling-----
    opt["sampler"]["num_sample_steps"] = nfe
    opt["sampler"]["time_schedule"] = "linear-lambda"
    opt["sampler"]["solver_type"] = "sde"
    opt["sampler"]["parameterization"] = parameterization
    sampler = MRSampler(sde=sde, opt=opt, **opt["sampler"])
    model.model.eval()
    with torch.no_grad():
        dpm_result, qbuffer, x_list1 = sampler.sampling(model.condition, text_context=degra_context, image_context=image_context, use_tqdm=True, savedir=savedir, noisy_state=noisy_state.to(device))

    # -----compute metrics-----
    def compute_metric(pred, gt):
        lpips_score = lpips_fn(pred*2-1, gt.to(device)*2-1).item()
        pred_img = tensor2img(pred)
        gt_img = tensor2img(gt)
        psnr = calculate_psnr(pred_img, gt_img)
        ssim = calculate_ssim(pred_img, gt_img)
        print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_score}")
        with open(os.path.join(savedir, "metric.txt"), "a") as f:
            f.write(f"MR Sampler-2-({nfe})-({parameterization}):\n")
            f.write(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_score}\n")
    cv2.imwrite(os.path.join(savedir,"gt.png"), tensor2img(GT))
    cv2.imwrite(os.path.join(savedir,"lq.png"), tensor2img(LQ))
    cv2.imwrite(os.path.join(savedir,"sample_result.png"), tensor2img(dpm_result))
    print(f"MR Sampler-2-({nfe})-({parameterization}):")
    compute_metric(dpm_result, GT)

    # -----check differential coefficient-----
    d0_ls = qbuffer[1:]
    d1_ls = []
    lambdas = sampler.lambdas
    h = lambdas[2:] - lambdas[1:-1]
    for i in range(1, len(qbuffer)):
        d1_ls.append((qbuffer[i] - qbuffer[i-1])/(lambdas[i+1] - lambdas[i]))
    R = [torch.nan_to_num(torch.abs(d0/d1)) for d0,d1 in zip(d0_ls, d1_ls)]
    ratio = [torch.mean((r>hi)*1.0).item() for r, hi in zip(R, h)]
    fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
    plt.bar(range(2,len(ratio)+2), ratio)
    # 使用 text 方法添加自定义标签
    ax.text(0.5, -0.1, 'Index of timestep', transform=ax.transAxes, ha='center', va='top', fontdict={'family': 'serif', 'color': 'black', 'weight': 'bold', 'size': 20})
    # ax.text(-0.1, 0.5, 'PC2', transform=ax.transAxes, ha='right', va='center', fontsize=18)
    # 隐藏默认的标签
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.title("Ratio of Convergence", fontdict={'family': 'serif', 'color': 'black', 'weight': 'bold', 'size': 25})
    plt.savefig(os.path.join(savedir, "ratio.png"))
    return ratio
    # Rmin = [torch.min(torch.nan_to_num(torch.abs(d0/d1))).item() for d0,d1 in zip(d0_ls, d1_ls)]
    # Rmean = [torch.mean(torch.nan_to_num(torch.abs(d0/d1))).item() for d0,d1 in zip(d0_ls, d1_ls)]
    # plt.figure()
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    # plt.plot(Rmin, marker='x', color='red', label='Rmin')
    # plt.plot(h[:-1], marker='o', color='blue', label='h')
    # plt.xlabel("index of timestep"), plt.legend()
    # plt.savefig(os.path.join(savedir, "Rmin.png"))
    # plt.figure()
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    # plt.plot(Rmean, marker='x', color='red', label="Rmean")
    # plt.plot(h[:-1], marker='o', color='blue', label="h")
    # plt.xlabel("index of timestep"), plt.legend()
    # plt.savefig(os.path.join(savedir, "Rmean.png"))

os.mkdir(os.path.join(savedir, "noise"))
ratio1 = analysis(15, "noise", os.path.join(savedir, "noise"))
with open(os.path.join(savedir, "noise", "metric.txt"), "a") as f:
    f.write(f"ratio of convergence {ratio1}\n")

os.mkdir(os.path.join(savedir, "data"))
ratio2 = analysis(15, "data", os.path.join(savedir, "data"))
with open(os.path.join(savedir, "data", "metric.txt"), "a") as f:
    f.write(f"ratio of convergence {ratio2}\n")
# %%
# 设置字体
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.plot(range(2,len(ratio1)+2), ratio1, linewidth=2, color='#293890', marker='x', markersize=10)
plt.plot(range(2,len(ratio2)+2), ratio2, linewidth=2, color='#BF1D2D', marker='o', markersize=10)
# 使用 text 方法添加自定义标签
ax.text(0.5, -0.1, 'Index of timestep', transform=ax.transAxes, ha='center', va='top', fontdict={'color': 'black', 'size': 18, 'weight': 'bold'})
# 隐藏默认的标签
ax.set_xlabel('')
ax.set_ylabel('')
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
plt.legend(["Noise prediction", "Data prediction"], loc='upper right', bbox_to_anchor=(1, 0.92), prop={'size': 18, 'weight': 'bold'})
# plt.title("Ratio of Convergence", fontdict={'family': 'serif', 'color': 'black', 'weight': 'bold', 'size': 25})
plt.savefig(os.path.join(savedir, "ratio.pdf"), bbox_inches='tight', format='pdf')
# %%
