# generate the intro figure in the paper
import torch
import sys, os
sys.path.append(os.path.join(os.getcwd(),"universal-image-restoration"))
sys.path.append(os.path.join(os.getcwd(),"universal-image-restoration/config/daclip-sde"))
import open_clip
from models import create_model
import options
from data import util
import cv2
from utils.sampler import MRSampler, HeunSampler, AdamsMoulton
from glob import glob
import lpips
from utils.img_utils import tensor2img, calculate_psnr, calculate_ssim
from utils.sde_utils import IRSDE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

workdir = os.getcwd()
savedir = os.path.join(workdir, "scripts/samples")
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
srcpath = glob("/mnt/intern/zhejing.la/data/4_daclip/daclip-val/uncompleted/LQ/*")[0]
tgtpath = glob("/mnt/intern/zhejing.la/data/4_daclip/daclip-val/uncompleted/GT/*")[0]
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
nfe_ls = [5, 20, 50, 80, 100]
opt["sampler"]["timestep_type"] = "linear-lambda"
opt["sampler"]["solver_type"] = "sde"
model.model.eval()
def compute_metric(pred, gt):
    lpips_score = lpips_fn(pred*2-1, gt.to(device)*2-1).item()
    pred_img = tensor2img(pred)
    gt_img = tensor2img(gt)
    psnr = calculate_psnr(pred_img, gt_img)
    ssim = calculate_ssim(pred_img, gt_img)
    print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_score}")

with torch.no_grad():
    for nfe in nfe_ls:
        sde.set_mu(model.condition)
        post_result = sde.reverse_posterior(model.state, num_steps=nfe, save_states=False, text_context=degra_context, image_context=image_context)
        
        opt["sampler"]["order"] = 2
        opt["sampler"]["num_sample_steps"] = nfe
        sampler = MRSampler(sde=sde, opt=opt, **opt["sampler"])
        mrs_result, qbuffer, x_list = sampler.sampling(model.condition, text_context=degra_context, image_context=image_context, use_tqdm=True, savedir=None, noisy_state=model.state)

        # -----compute metrics-----
        cv2.imwrite(os.path.join(savedir,"lq.png"), tensor2img(LQ))
        cv2.imwrite(os.path.join(savedir,"gt.png"), tensor2img(GT))
        cv2.imwrite(os.path.join(savedir,f"post_result({nfe}).png"), tensor2img(post_result))
        cv2.imwrite(os.path.join(savedir,f"mrs_result({nfe}).png"), tensor2img(mrs_result))
        print(f"posterior({nfe})")
        compute_metric(post_result, GT)
        print(f"MRS2({nfe})")
        compute_metric(mrs_result, GT)

