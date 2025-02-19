import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import open_clip
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
try:
    from utils import OrderedYaml
except ImportError:
    pass
import yaml
Loader, Dumper = OrderedYaml()

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
parser.add_argument("--sampler", type=str, default=None, help="Sampling method", choices=["mrsampler", "sde", "posterior", "heun"])
parser.add_argument("--nfe", type=int, default=None, help="Number of function evaluations")
parser.add_argument("--dataset", type=str, default=None, help="which dataset")

args = parser.parse_args()
with open(args.opt, "r") as f:
    opt = yaml.load(f, Loader=Loader)
if args.sampler is not None:
    opt["sde"]["algorithm"] = args.sampler
    if args.sampler == "posterior" or args.sampler == "sde":
        opt["name"] = "universal-ir"
    elif args.sampler == "mrsampler":
        opt["name"] = "universal-ir-mrsampler"
    else:
        opt["name"] = "universal-ir-heun"
if args.nfe is not None:
    opt["sampler"]["num_sample_steps"] = args.nfe
if args.dataset is not None:
    opt["datasets"]["test1"]["name"] = args.dataset
    opt["datasets"]["test1"]["dataroot_GT"] = f"/mnt/intern/zhejing.la/data/4_daclip/daclip-val/{args.dataset}/GT"
    opt["datasets"]["test1"]["dataroot_LQ"] = f"/mnt/intern/zhejing.la/data/4_daclip/daclip-val/{args.dataset}/LQ"
opt = option.parse(opt_path="", opt=opt, is_train=False)
opt = option.dict_to_nonedict(opt)


#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

# clip_model, _preprocess = clip.load("ViT-B/32", device=device)
if opt['path']['daclip'] is not None:
    clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
else:
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model = clip_model.to(device)


sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
lpips_fn = lpips.LPIPS(net='alex').to(device)

scale = opt['degradation']['scale']
algorithm = opt["sde"]["algorithm"]
if algorithm == "mrsampler":
    sampler = util.sampler.MRSampler(
        sde=sde,
        opt=opt,
        **opt["sampler"],
    )
elif algorithm == "heun":
    sampler = util.sampler.HeunSampler(
        sde=sde,
        nfe=opt["sampler"]["num_sample_steps"],
        solver_type=opt["sampler"]["solver_type"],
    )

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = []
    test_times = []

    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ, GT = test_data["LQ"], test_data["GT"]
        img4clip = test_data["LQ_clip"].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_context, degra_context = clip_model.encode_image(img4clip, control=True)
            image_context = image_context.float()
            degra_context = degra_context.float()

        noisy_state = sde.noise_state(LQ)

        model.feed_data(noisy_state, LQ, GT, text_context=degra_context, image_context=image_context)
        tic = time.time()
        if algorithm == "mrsampler":
            model.model.eval()
            model.output, _, _ = sampler.sampling(model.condition, text_context=degra_context, image_context=image_context, use_tqdm=True)
        elif algorithm == "heun":
            model.model.eval()
            model.output, _ = sampler.sampling(model.condition, text_context=degra_context, image_context=image_context)
        else:
            model.model.eval()
            model.test(sde, mode=algorithm, num_steps=opt["sampler"]["num_sample_steps"], save_states=False)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())  # uint8
        LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8
        
        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
        util.save_img(output, save_img_path)

        # remove it if you only want to save output images
        # LQ_img_path = os.path.join(dataset_dir, img_name + "_LQ.png")
        # GT_img_path = os.path.join(dataset_dir, img_name + "_HQ.png")
        # util.save_img(LQ_, LQ_img_path)
        # util.save_img(GT_, GT_img_path)

        if need_GT:
            gt_img = GT_ / 255.0
            sr_img = output / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else scale
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                cropped_gt_img = gt_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            lp_score = lpips_fn(
                GT.to(device) * 2 - 1, SR_img.to(device) * 2 - 1).squeeze().item()

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lp_score)

            if len(gt_img.shape) == 3:
                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_border == 0:
                        cropped_sr_img_y = sr_img_y
                        cropped_gt_img_y = gt_img_y
                    else:
                        cropped_sr_img_y = sr_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                        cropped_gt_img_y = gt_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                    psnr_y = util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    ssim_y = util.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )

                    test_results["psnr_y"].append(psnr_y)
                    test_results["ssim_y"].append(ssim_y)

                    logger.info(
                        "img{:3d}:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                            i, img_name, psnr, ssim, lp_score, psnr_y, ssim_y
                        )
                    )
            else:
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                        img_name, psnr, ssim
                    )
                )

                test_results["psnr_y"].append(psnr)
                test_results["ssim_y"].append(ssim)
        else:
            logger.info(img_name)


    ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )
    if test_results["psnr_y"] and test_results["ssim_y"]:
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
        logger.info(
            "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                ave_psnr_y, ave_ssim_y
            )
        )

    logger.info(
            "----average LPIPS\t: {:.6f}\n".format(ave_lpips)
        )

    logger.info(f"average test time: {np.mean(test_times):.4f} s")

    # FID
    import torch
    import os
    from pytorch_fid.fid_score import save_fid_stats, calculate_fid_given_paths

    paths = [
        dataset_dir,
        opt["datasets"]["test1"]["dataroot_GT"]
    ]
    batch_size = 1
    dims = 2048
    save_stats = False
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    num_workers = 4

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity is not available under Windows, use
        # os.cpu_count instead (which may not return the *available* number
        # of CPUs).
        num_cpus = os.cpu_count()
    num_workers = min(num_cpus, num_workers) if num_cpus is not None else 1

    if save_stats:
        save_fid_stats(paths, batch_size, device, dims, num_workers)
        exit()

    fid_value = calculate_fid_given_paths(
        paths, batch_size, device, dims, num_workers
    )
    logger.info(f"---average FID: {fid_value}")

    # output metrics to excel
    import pandas as pd
    filepath = os.path.join(opt["path"]["results_root"], test_set_name+".xlsx")
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        new_rows = pd.DataFrame({
            "sampler": [algorithm],
            "NFE": [opt["sampler"]["num_sample_steps"]],
            "PSNR": [ave_psnr],
            "SSIM": [ave_ssim],
            "LPIPS": [ave_lpips],
            "PSNR-Y": [ave_psnr_y],
            "SSIM-Y": [ave_ssim_y],
            "FID": [fid_value],
            "time": [np.mean(test_times)]
        })
        df = pd.concat([df, new_rows], ignore_index=True)
    else:
        df = pd.DataFrame({
            "sampler": [algorithm],
            "NFE": [opt["sampler"]["num_sample_steps"]],
            "PSNR": [ave_psnr],
            "SSIM": [ave_ssim],
            "LPIPS": [ave_lpips],
            "PSNR-Y": [ave_psnr_y],
            "SSIM-Y": [ave_ssim_y],
            "FID": [fid_value],
            "time": [np.mean(test_times)]
        })
    df.to_excel(filepath, index=False, engine='openpyxl')
    # 使用 xlsxwriter 输出到 Excel
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)

        # 获取 xlsxwriter Workbook 和 Worksheet 对象
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # 设置浮点数小数位
        format_float = workbook.add_format({'num_format': '0.000000000'})

        # 应用格式到列
        worksheet.set_column('C:I', None, format_float)  # 设置第 B 列