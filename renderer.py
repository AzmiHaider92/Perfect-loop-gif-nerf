import numpy as np
import torch,os,imageio,sys
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from TensoRF.camera.visualize_positions import points_from_transforms
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimsave(f'{savePath}/{prtx}video.gif', np.stack(rgb_maps), fps=20, format='GIF')
    imageio.mimsave(f'{savePath}/{prtx}depthvideo.gif', np.stack(depth_maps), fps=20, format='GIF')

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, image_paths, added_image_indices, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', added=0):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):
        # LOCATIONS MAP
        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(10,5), gridspec_kw={'width_ratios': [2, 1.5]})
        ax.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        ax2.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        pts = points_from_transforms(c2ws)
        pts = pts[:, [0, 1, 3]]
        ax2.scatter(pts[:, 0], pts[:, 1], color='red')
        idxs = [i for i in range(len(added_image_indices)) if added_image_indices[i]]
        ax2.scatter(pts[idxs, 0], pts[idxs, 1], color='green')
        ax2.scatter(0, 0, color='black', marker='x', s=200)


        W, H = test_dataset.img_wh
        if added_image_indices[idx]:
            c2w = torch.FloatTensor(np.array(c2w.data))
            rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
            if ndc_ray:
                rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

            rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                                   ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            rgb_map = rgb_map.clamp(0.0, 1.0)

            rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

            depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

            rgb_map = (rgb_map.numpy() * 255).astype('uint8')
            # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            #height, width, _ = rgb_map.shape
            #newsize = (int(width / 5), int(height / 5))
            #rgb_map = cv2.resize(rgb_map, newsize)
            # green circle in image
            #rgb_map = cv2.circle(rgb_map, (10, 10), 10, (0, 255, 0), -1)
            rgb_map = create_border(rgb_map, 20, np.array([0, 255,  0]))
            ax2.scatter(pts[idx, 0], pts[idx, 1], color='green', marker='o', s=200)
        else:
            # blue circle
            rgb_map = cv2.imread(image_paths[idx])
            rgb_map = cv2.cvtColor(rgb_map, cv2.COLOR_BGR2RGB)
            rgb_map = create_border(rgb_map,20,np.array([255, 0, 0]))
            ax2.scatter(pts[idx, 0], pts[idx, 1], color='red', marker='o', s=200)
            #rgb_map = cv2.circle(rgb_map, (10, 10), 10, (255, 0, 0), -1)

        ax.imshow(rgb_map)
        #depth_maps.append(depth_map)
        if savePath is not None:
            plt.savefig(f'{savePath}/{prtx}{idx:03d}.png')
            #imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            #rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            #imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    #imageio.mimsave(f'{savePath}/{prtx}video.gif', np.stack(rgb_maps), fps=20, format='GIF')
    #imageio.mimsave(f'{savePath}/{prtx}depthvideo.gif', np.stack(depth_maps), fps=20, format='GIF')

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs


def create_border(img, width, color=np.array([0, 0, 0])):
    #color must be a np.array

    img_shape = img.shape
    upper_border = np.full((width, img_shape[1], 3), color) #for 3-channel image
    side_border = np.full((img_shape[0] + 2*width, width, 3), color)

    bordered = np.concatenate([upper_border, img, upper_border])
    bordered = np.concatenate([side_border, bordered, side_border], axis=1)

    return bordered

