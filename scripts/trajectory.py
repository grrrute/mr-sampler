# %%
import torch
import sys, os
workdir = os.getcwd()
sys.path.append(os.path.join(workdir,"universal-image-restoration"))
sys.path.append(os.path.join(workdir,"universal-image-restoration/config/daclip-sde"))
import open_clip
from models import create_model
import options
from data import util
import cv2
from utils.sampler import MRSampler, HeunSampler, AdamsMoulton
from utils.sde_utils import IRSDE
from glob import glob
import lpips
from utils.img_utils import tensor2img, calculate_psnr, calculate_ssim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import seaborn

savedir = os.path.join(workdir,"scripts/trajectory")
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
# get test data, set the following path as your own path to dataset
srcpath = glob("/mnt/intern/zhejing.la/data/4_daclip/daclip-val/motion-blurry/LQ/*")[12]
tgtpath = glob("/mnt/intern/zhejing.la/data/4_daclip/daclip-val/motion-blurry/GT/*")[12]
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

# -----sampling-----
opt["sampler"]["num_sample_steps"] = 10
opt["sampler"]["time_schedule"] = "linear-lambda"
opt["sampler"]["solver_type"] = "sde"
model.model.eval()

with torch.no_grad():
    sde.set_mu(model.condition)
    gt_result, x_list1 = sde.reverse_posterior(model.state, num_steps=100, save_states=True, text_context=degra_context, image_context=image_context)

    post_result, x_list2 = sde.reverse_posterior(model.state, num_steps=opt["sampler"]["num_sample_steps"], save_states=True, text_context=degra_context, image_context=image_context)
    
    euler_result, x_list3 = sde.reverse_sde(model.state, num_steps=opt["sampler"]["num_sample_steps"], save_states=True, text_context=degra_context, image_context=image_context)

    opt["sampler"]["order"] = 1
    sampler1 = MRSampler(sde=sde, opt=opt, **opt["sampler"])
    dpm1_result, qbuffer, x_list4 = sampler1.sampling(model.condition, text_context=degra_context, image_context=image_context, use_tqdm=True, savedir=savedir, noisy_state=model.state)

    opt["sampler"]["order"] = 2
    sampler2 = MRSampler(sde=sde, opt=opt, **opt["sampler"])
    dpm2_result, qbuffer, x_list5 = sampler2.sampling(model.condition, text_context=degra_context, image_context=image_context, use_tqdm=True, savedir=savedir, noisy_state=model.state)


# adamsampler = AdamsMoulton(sde=sde, order=4, nfe=opt["sampler"]["num_sample_steps"]*2)
# print(adamsampler.timesteps)
# with torch.no_grad():
#     adam_result, x_list3 = adamsampler.sampling(model.condition, text_context=degra_context, image_context=image_context, noisy_state=model.state)


# -----compute metrics-----
def compute_metric(pred, gt):
    lpips_score = lpips_fn(pred*2-1, gt.to(device)*2-1).item()
    pred_img = tensor2img(pred)
    gt_img = tensor2img(gt)
    psnr = calculate_psnr(pred_img, gt_img)
    ssim = calculate_ssim(pred_img, gt_img)
    print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_score}")
cv2.imwrite(os.path.join(savedir,"lq.png"), tensor2img(LQ))
cv2.imwrite(os.path.join(savedir,"gt.png"), tensor2img(GT))
cv2.imwrite(os.path.join(savedir,"post_result.png"), tensor2img(post_result))
cv2.imwrite(os.path.join(savedir,"euler_result.png"), tensor2img(euler_result))
cv2.imwrite(os.path.join(savedir,"dpm1_result.png"), tensor2img(dpm1_result))
cv2.imwrite(os.path.join(savedir,"dpm2_result.png"), tensor2img(dpm2_result))
print("posterior")
compute_metric(post_result, GT)
print("euler")
compute_metric(euler_result, GT)
print("DPM1")
compute_metric(dpm1_result, GT)
print("DPM2")
compute_metric(dpm2_result, GT)

# -----pca------
def get_pca_points(x_list, y_list):
    pca = PCA(n_components=2)
    X = torch.stack(x_list).reshape(len(x_list), -1).numpy()
    Y = torch.stack(y_list).reshape(len(y_list), -1).numpy()
    pca.fit(X)
    points = pca.transform(Y)
    print(f"pca explained variance ratio: {pca.explained_variance_ratio_}")
    return points
points1 = get_pca_points([GT.cpu()]+x_list1+x_list2+x_list3+x_list4+x_list5, x_list1)
points2 = get_pca_points([GT.cpu()]+x_list1+x_list2+x_list3+x_list4+x_list5, x_list2)
points3 = get_pca_points([GT.cpu()]+x_list1+x_list2+x_list3+x_list4+x_list5, x_list3)
points4 = get_pca_points([GT.cpu()]+x_list1+x_list2+x_list3+x_list4+x_list5, x_list4)
points5 = get_pca_points([GT.cpu()]+x_list1+x_list2+x_list3+x_list4+x_list5, x_list5)
gt_point = get_pca_points([GT.cpu()]+x_list1+x_list2+x_list3+x_list4+x_list5, [GT.cpu()])[0]

# %%
# -----draw-----
def draw_traj(points, ax, color:str, marker, arrowsize=20, linewidth=2, stride:int=1):
    # 绘制每一步的箭头, 每stride步绘制一次
    for i in range(len(points) - 1):
        if i%stride == 0:
            start_point = points[i]
            end_point = points[i + stride]
            ax.annotate('', xy=end_point, xytext=start_point, arrowprops=dict(arrowstyle='->', color=color, linewidth=linewidth, mutation_scale=arrowsize))
    # 绘制终点
    ax.scatter(points[-1][0], points[-1][1], c=color, s=100, marker=marker)
    # 设置坐标轴范围和比例
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(min([x for x, y in points]+[xmin])-1, max([x for x, y in points]+[xmax]) + 1)
    ax.set_ylim(min([y for x, y in points]+[ymin])-1, max([y for x, y in points]+[ymax]) + 1)
    
# 设置字体
# plt.rcParams['font.family'] = 'Latin Modern Math'
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.figure(figsize=(12,10)), plt.gca()
ax.scatter(gt_point[0], gt_point[1], c="#FF0000", s=1000, marker='*')

draw_traj(points1, ax, '#808080','o',30)
draw_traj(points2, ax, '#589FF3','o',30)
draw_traj(points3, ax, '#F3B169','o',30)
draw_traj(points4, ax, '#F94141','o',30)
draw_traj(points5, ax, '#37AB78','o',30)

plt.legend(["GroundTruth", "Posterior Sampling(100)", "Posterior Sampling(10)", "Euler Discretization(10)", "MR Sampler-1(10)", "MR Sampler-2(10)"], prop={'size': 20, 'weight': 'bold'})
# 使用 text 方法添加自定义标签
ax.text(0.5, -0.07, 'PC1', transform=ax.transAxes, ha='center', va='top', fontsize=24, fontdict={'weight': 'bold'})
ax.text(-0.08, 0.5, 'PC2', transform=ax.transAxes, ha='right', va='center', fontsize=24, fontdict={'weight': 'bold'}, rotation=90)
# 隐藏默认的标签
ax.set_xlabel('')
ax.set_ylabel('')
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.savefig(os.path.join(savedir, "traj.pdf"), bbox_inches='tight', format='pdf')
# %%
