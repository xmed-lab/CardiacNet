# CUDA_VISIBLE_DEVICES=1,2,4,5 
import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
# from transformer import VQGANTransformer
from time_vqgan_template import VQGAN
from utils import load_data, plot_images
from data.cardiacnet import CardiacNet
from monai.data import DataLoader
import cv2 as cv

from scipy import linalg

from einops import rearrange

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects
from matplotlib import pyplot as plt 
import matplotlib

from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
import torchmetrics

# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X, y)
# Pipeline(steps=[('standardscaler', StandardScaler()),
#                 ('svc', SVC(gamma='auto'))])

view_select = '4'
epoch_select = '200'

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv3d(args.latent_dim, args.latent_dim * 2, kernel_size=3, stride=2, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(args.latent_dim * 2, 1)

    def forward(self, input):
        x = self.conv1(input)
        # x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = x
        x = self.fc(x)
        # x = F.sigmoid(x)
        return x, feat

class Test:
    def __init__(self, args):
        loaded_dict = {}
        self.model_normal = VQGAN(args).to(device=args.device)
        args.checkpoint_path = f"./checkpoints/vqgan_dual_normal_epoch_{epoch_select}.pt".format()
        # args.checkpoint_path = f"./view4/new_checkpoints/vqgan_dual_normal_epoch_55.pt"
        self.model_normal.load_state_dict({k.replace('module.',''):v for k, v in torch.load(args.checkpoint_path, map_location=args.device).items()})
        
        # for k, v in torch.load(args.checkpoint_path, map_location=args.device).items():
        #     if 'encoder' in k:
        #         loaded_dict.update({k.replace('module.',''):v})
        
        # args.checkpoint_path = r"./checkpoints/vqgan_dual_normal_epoch_114.pt"
        # #self.model_normal.load_state_dict({k.replace('module.',''):v for k, v in torch.load(args.checkpoint_path, map_location=args.device).items()})
        # for k, v in torch.load(args.checkpoint_path, map_location=args.device).items():
        #     if 'encoder' not in k:
        #         loaded_dict.update({k.replace('module.',''):v})
        
        # print(loaded_dict.keys())
        # self.model_normal.load_state_dict(loaded_dict)

        self.model_abnorm = VQGAN(args).to(device=args.device)
        args.checkpoint_path = f"./checkpoints/vqgan_dual_abnorm_epoch_{epoch_select}.pt"
        # args.checkpoint_path = f"./view4/new_checkpoints/vqgan_dual_abnorm_epoch_55.pt"
        self.model_abnorm.load_state_dict({k.replace('module.',''):v for k, v in torch.load(args.checkpoint_path, map_location=args.device).items()})

        self.FID_model = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True).to(device=args.device)
        self.FID_model.eval()

        infos = np.load(f'/home/jyangcu/Dataset/dataset_pa_iltrasound_nii_files_3rdcenters/save_infos_reg_v3.npy', allow_pickle=True).item()
        # 'ASD-severe' 'normal' 'middle' 'severe'
        train_abnorm_dataset = CardiacNet(args, infos, set_select=['gy','rmyy','shph'], view_num=[view_select], data_type=['ASD-severe'], is_train=True, is_test=False)
        train_abnorm = DataLoader(train_abnorm_dataset, batch_size=args.batch_size * 4, shuffle=True, num_workers=4)
        train_normal_dataset = CardiacNet(args, infos, set_select=['gy','rmyy','shph'], view_num=[view_select], data_type=['normal'], is_train=True, is_test=False)
        train_normal = DataLoader(train_normal_dataset, batch_size=args.batch_size * 4, shuffle=True, num_workers=4)

        test_abnorm_dataset = CardiacNet(args, infos, set_select=['gy','rmyy','shph'], view_num=[view_select], data_type=['ASD-severe'], is_train=True, is_test=True)
        test_abnorm = DataLoader(test_abnorm_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_normal_dataset = CardiacNet(args, infos, set_select=['gy','rmyy','shph'], view_num=[view_select], data_type=['normal'], is_train=True, is_test=True)
        test_normal = DataLoader(test_normal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        #self.train_svm(args, train_abnorm, train_normal, test_normal, test_abnorm)
        self.test(args, test_abnorm, test_normal)

    def train_svm(self, args, train_abnorm, train_normal, test_normal, test_abnorm):

        classifier = Classifier(args).to(device=args.device)
        opt_classifier = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, eps=1e-08)
        bce_loss = nn.BCEWithLogitsLoss()

        self.model_normal.eval()
        self.model_abnorm.eval()
        steps_per_epoch = len(train_abnorm)

        gt_vids_feat = []
        re_vids_feat = []

        for epoch in range(args.epochs):
            train_normal_iter = iter(train_normal)
            features = []
            tgts = []
            features_test = []
            tgt_test = []
            with tqdm(range(steps_per_epoch)) as pbar:
                for i, (abnorm_vids, abnorm_masked_pos, _, _) in zip(pbar, train_abnorm):
                    normal_vids, normal_masked_pos, _, _ = next(train_normal_iter)

                    normal_vids = normal_vids.to(device=args.device)
                    abnorm_vids = abnorm_vids.to(device=args.device)
                    tgt = torch.cat([torch.zeros([args.batch_size * 4]), torch.ones([args.batch_size * 4])]).to(device=args.device)
                    with torch.no_grad():
                        _, _, _, (vq_feat_normal, _, _), feat_normal = self.model_normal(normal_vids, None)
                        _, _, _, (vq_feat_abnorm, _, _), feat_abnorm = self.model_normal(abnorm_vids, None)
                    feats = torch.cat([feat_normal, feat_abnorm])
                    output, feat = classifier(feats)
                    if i > 1000:
                        features.append(feat.detach().cpu())
                        tgts.append(tgt.detach().cpu())

                    if i % 100 == 0 and i > 1000:
                        features_tsne = torch.cat(features, dim=0)
                        tgts_tsne = torch.cat(tgts, dim=0)
                        digits_proj = TSNE(n_components=2, init='pca', random_state=666).fit_transform(features_tsne)
                        self.scatter(digits_proj, tgts_tsne, type='train')

                    loss = bce_loss(output.squeeze(-1), tgt)
                    opt_classifier.zero_grad()
                    loss.backward()
                    opt_classifier.step()

                    if i % 1000 == 0 and i > 1000:
                        print('Testing')
                        classifier.eval()
                        preds = []
                        gts = []
                        with torch.no_grad():
                            for i, (t_normal_vids, t_normal_masked_pos, _, _) in enumerate(test_normal):
                                t_normal_vids = t_normal_vids.to(device=args.device)
                                _, _, _, (t_normal_vq_feats, _, _), t_normal_feats = self.model_normal(t_normal_vids, None)
                                pred, feat_norm_test = classifier(t_normal_feats)
                                # pred = torch.where(pred > 0.5, 1, 0)
                                preds.append(pred.squeeze(-1).cpu())
                                gts.append(torch.zeros([args.batch_size]))
                                features_test.append(feat_norm_test.detach().cpu())
                                tgt_test.append(torch.zeros([args.batch_size]))
                            
                            for i, (t_abnorm_vids, t_abnorm_masked_pos, _, _) in enumerate(test_abnorm):
                                t_abnorm_vids = t_abnorm_vids.to(device=args.device)
                                _, _, _, (t_abnorm_vq_feats, _, _), t_abnorm_feats = self.model_normal(t_abnorm_vids, None)
                                pred, feat_abno_test = classifier(t_abnorm_feats)
                                # pred = torch.where(pred > 0.5, 1, 0)
                                preds.append(pred.squeeze(-1).cpu())
                                gts.append(torch.ones([args.batch_size]))
                                features_test.append(feat_abno_test.detach().cpu())
                                tgt_test.append(torch.ones([args.batch_size]))
                            
                            gts = torch.cat(gts).to(torch.int32)
                            preds = torch.cat(preds)
                            auroc_metric = BinaryAUROC(thresholds=None)
                            print("auc_roc", auroc_metric(preds, gts))
                                
                            confmat = torchmetrics.ConfusionMatrix(task="binary")
                            roc = torchmetrics.ROC(task="binary")
                            collection = torchmetrics.MetricCollection(
                                        torchmetrics.Accuracy(task="binary"),
                                        torchmetrics.Recall(task="binary"),
                                        torchmetrics.Precision(task="binary"),
                                        confmat,
                                        roc,
                                    )
                            # Define tracker over the collection to easy keep track of the metrics over multiple steps
                            tracker = torchmetrics.wrappers.MetricTracker(collection)
                            tracker.increment()
                            # Run "training" loop
                            tracker.update(preds, gts)

                            # Extract all metrics from all steps
                            all_results = tracker.compute_all()

                            # Constuct a single figure with appropriate layout for all metrics
                            # fig = plt.subplots(layout="constrained")
                            ax1 = plt.subplot(2, 2, 1)
                            ax2 = plt.subplot(2, 2, 2)
                            # ax3 = plt.subplot(2, 2, (3, 4))

                            # ConfusionMatrix and ROC we just plot the last step, notice how we call the plot method of those metrics
                            confmat.plot(val=all_results[-1]['BinaryConfusionMatrix'], ax=ax1)
                            roc.plot(all_results[-1]["BinaryROC"], ax=ax2)
                            plt.savefig("test.png")
                            # For the remainig we plot the full history, but we need to extract the scalar values from the results
                            # scalar_results = [
                            #     {k: v for k, v in ar.items() if isinstance(v, torch.Tensor) and v.numel() == 1} for ar in all_results
                            # ]
                            # tracker.plot(val=scalar_results, ax=ax3)

                            acc_metric = BinaryAccuracy()
                            print("acc", acc_metric(preds, gts))

                            print("visulize TSNE")
                            features_test_tsne = torch.cat(features_test, dim=0)
                            tgt_test_tsne = torch.cat(tgt_test, dim=0)
                            digits_proj = TSNE(n_components=2, init='pca', random_state=666).fit_transform(features_test_tsne)
                            self.scatter(digits_proj, tgt_test_tsne, type='test')
                        
                        classifier.train()
                    if i > 1500:
                        break


    def test(self, args, test_abnorm, test_normal):
        self.model_normal.eval()
        self.model_abnorm.eval()
        steps_per_epoch = len(test_abnorm)

        gt_vids_feat = []
        re_vids_feat = []

        for epoch in range(args.epochs):
            test_normal_iter = iter(test_normal)
            with tqdm(range(steps_per_epoch)) as pbar:
                for i, (abnorm_vids, abnorm_masked_pos, abnorm_mpap, abnorm_pasp) in zip(pbar, test_abnorm):
                    #normal_vids, normal_masked_pos, normal_mpap, normal_pasp = next(test_normal_iter)

                    abnorm_masked_pos = rearrange(abnorm_masked_pos.to(device=args.device), 'b (l h w) p1 p2 -> b l (h p1) (w p2)', 
                                                  h=args.image_size[0]//args.mask_size, w=args.image_size[1]//args.mask_size, 
                                                  l=args.image_size[2], p1=args.mask_size, p2=args.mask_size).unsqueeze(1)
                    #normal_vids = normal_vids.to(device=args.device)
                    abnorm_vids = abnorm_vids.to(device=args.device)
                    sampled_vids_abnorm_to_abnorm, _, _, _, _ = self.model_normal(abnorm_vids, None, type = 'N2A')
                    sampled_vids_abnorm_to_normal, _, _, _, _ = self.model_abnorm(abnorm_vids, None, type = 'A2N')

                    images_normal, images_recon_normal, heatmaps_normal = self.convert_heatmap(abnorm_vids, sampled_vids_abnorm_to_normal, device=args.device)
                    sampled_imgs_normal = torch.cat((images_normal.add(1.0).mul(0.5),
                                              images_recon_normal.add(1.0).mul(0.5),
                                              heatmaps_normal.add(1.0).mul(0.5),))
                    vutils.save_image(sampled_imgs_normal, os.path.join(f"view{view_select}/results/transform", f"transformer_abnormal2normal_{i}.jpg"), nrow=args.image_size[2])       

                    images_abnorm, images_recon_abnorm, heatmaps_abnorm = self.convert_heatmap(abnorm_vids, sampled_vids_abnorm_to_abnorm, device=args.device, type='abnorm')
                    sampled_imgs_abnorm = torch.cat((images_abnorm.add(1.0).mul(0.5),
                                              images_recon_abnorm.add(1.0).mul(0.5),
                                              heatmaps_abnorm.add(1.0).mul(0.5),))
                    vutils.save_image(sampled_imgs_abnorm, os.path.join(f"view{view_select}/results/transform", f"transformer_abnormal2abnormal_{i}.jpg"), nrow=args.image_size[2])
                    # if i >= 100:
                    #     break

                    gt_feat, re_feat = self.compute_feature(abnorm_vids, sampled_vids_abnorm_to_normal)
                    gt_vids_feat.append(gt_feat.detach())
                    re_vids_feat.append(re_feat.detach())
                
                gt_vids_feat = torch.cat(gt_vids_feat, dim=0)
                re_vids_feat = torch.cat(re_vids_feat, dim=0)

                print("FID Score is", self.compute_FID_scroe(gt_vids_feat, re_vids_feat))


    def convert_heatmap(self, x, recon_x, device, type='normal'):

        def filter(img):
            img = cv.GaussianBlur(img, (5, 5), 0)
            img = cv.bilateralFilter(img, 9, 25, 25)
            return img
        
        x = x.add(1.0).mul(127.5).detach().cpu().squeeze().unsqueeze(-1).numpy()
        recon_x = recon_x.add(1.0).mul(127.5).detach().cpu().squeeze().unsqueeze(-1).numpy()

        images = list()
        images_recon = list()
        heatmaps = list()
        for idx, img_x in enumerate(x):
            #img_x = cv.flip(img_x, 1)
            #img_recon = cv.flip(img_recon, 1)
            img_recon = recon_x[idx]
            img_x_blur = img_x
            img_recon_blur = recon_x[idx]
            # img_x_blur = filter(img_x)
            # img_recon_blur = filter(recon_x[idx])

            if type == 'normal':
                img_x_blur = filter(img_x_blur)
                img_recon_blur = filter(img_recon_blur)
                img_x_blur = cv.dilate(img_x_blur, np.ones((5, 5), dtype=np.uint8), 3)
                img_x_blur = cv.erode(img_x_blur, np.ones((3, 3), dtype=np.uint8), iterations=1)
                img_recon_blur = cv.dilate(img_recon_blur, np.ones((5, 5), dtype=np.uint8), 3)
                img_recon_blur = cv.erode(img_recon_blur, np.ones((3, 3), dtype=np.uint8), iterations=1)
            else:
                # img_x_blur = cv.dilate(img_x_blur, np.ones((5, 5), dtype=np.uint8), iterations=1)
                img_x_blur = cv.erode(img_x_blur, np.ones((3, 3), dtype=np.uint8), iterations=1)
                # img_recon_blur = cv.dilate(img_recon_blur, np.ones((5, 5), dtype=np.uint8), iterations=1)
                img_recon_blur = cv.erode(img_recon_blur, np.ones((3, 3), dtype=np.uint8), iterations=1)

            img_difference = np.abs(img_x_blur - img_recon_blur)
            img_difference = ((img_difference - np.min(img_difference)) / np.max(img_difference))
            if type == 'normal':
                img_difference = np.where(img_difference > 0.30, img_difference, 0) * 255
            else:
                img_difference = np.where(img_difference > 0.20, img_difference, 0) * 255
            # img_difference = ((img_difference - np.min(img_difference)) / np.max(img_difference))
            img_difference = cv.bilateralFilter(img_difference, 9, 75, 75)
            # ret, thresh = cv.threshold(np.uint8(img_difference), 0, 255, cv.THRESH_BINARY)
            num_label, labels, stats, centroids = cv.connectedComponentsWithStats(img_difference.astype(np.uint8), connectivity=8)

            areas = list()
            for i in range(num_label):
                areas.append(stats[i][-1])
            
            area_avg = np.average(areas[1:-1])
            image_flitered = np.zeros_like(img_difference)
            for (i, label) in enumerate(np.unique(labels)):
                if label == 0:
                    continue
                if stats[i][-1] > 600:
                    image_flitered[labels == i] = img_difference[labels == i]
            
            img_x = cv.cvtColor(img_x, cv.COLOR_GRAY2RGB)
            images.append(torch.from_numpy(img_x).transpose(-1, 0).unsqueeze(0).to(device))
            img_recon = cv.cvtColor(img_recon, cv.COLOR_GRAY2RGB)
            images_recon.append(torch.from_numpy(img_recon).transpose(-1, 0).unsqueeze(0).to(device))

            image_flitered = cv.applyColorMap(image_flitered.astype(np.uint8), cv.COLORMAP_JET)
            # image_flitered = cv.GaussianBlur(image_flitered, (5, 5), 0)
            image_flitered = cv.cvtColor(image_flitered, cv.COLOR_BGR2RGB)
            image_flitered = img_x * 0.500 + image_flitered * 0.500
            # image_flitered = np.where(image_flitered == 0, img_x, img_x * 0.4 + image_flitered * 0.6)
            # image_flitered = cv.flip(image_flitered, 1)
            image_flitered = torch.from_numpy(image_flitered).transpose(-1, 0).unsqueeze(0).to(device)
            heatmaps.append(image_flitered)

        return torch.cat(images, dim=0).to(device).div(127.5)-1.0, torch.cat(images_recon, dim=0).to(device).div(127.5)-1.0, torch.cat(heatmaps, dim=0).to(device).div(127.5)-1.0

    def compute_feature(self, gt_vid, re_vid):

        def subtract_mean(x: torch.Tensor) -> torch.Tensor:
            mean = [0.406, 0.456, 0.485]
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            return x

        def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
            return x.mean([0, 2, 3]).view(1, -1)
                
        def get_feature(image):
            # If input has just 1 channel, repeat channel to have 3 channels
            if image.shape[1]:
                image = image.repeat(1, 3, 1, 1)

            # Change order from 'RGB' to 'BGR'
            image = image[:, [2, 1, 0], ...]

            # Subtract mean used during training
            image = subtract_mean(image)

            # Get model outputs
            with torch.no_grad():
                feature_image = self.FID_model.forward(image)
                # feature_image = spatial_average(feature_image, keepdim=False)

            return feature_image
    
        gt_vid = gt_vid.add(1.0).mul(0.5).detach().squeeze().unsqueeze(1)
        re_vid = re_vid.add(1.0).mul(0.5).detach().squeeze().unsqueeze(1)

        gt_features = []
        re_features = []
        for idx, gt_frame in enumerate(gt_vid):
            re_frame = re_vid[idx]
            gt_features.append(get_feature(gt_frame))
            re_features.append(get_feature(re_frame))
        return spatial_average(torch.cat(gt_features, dim=0)), spatial_average(torch.cat(re_features, dim=0))

    def compute_FID_scroe(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        gt = gt.double()
        pred = pred.double()

        if gt.ndimension() > 2:
            raise ValueError("Inputs should have (number images, number of features) shape.")

        mu_gt = torch.mean(gt, dim=0)
        mu_pred = torch.mean(pred, dim=0)
        sigma_gt = self._cov(gt, rowvar=False)
        sigma_pred = self._cov(pred, rowvar=False)

        return self.compute_frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)

    def compute_frechet_distance(self, mu_x: torch.Tensor, sigma_x: torch.Tensor, mu_y: torch.Tensor, sigma_y: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """The Frechet distance between multivariate normal distributions."""
        diff = mu_x - mu_y

        covmean = self._sqrtm(sigma_x.mm(sigma_y))

        # Product might be almost singular
        if not torch.isfinite(covmean).all():
            print(f"FID calculation produces singular product; adding {epsilon} to diagonal of covariance estimates")
            offset = torch.eye(sigma_x.size(0), device=mu_x.device, dtype=mu_x.dtype) * epsilon
            covmean = self._sqrtm((sigma_x + offset).mm(sigma_y + offset))

        # Numerical error might give slight imaginary component
        if torch.is_complex(covmean):
            if not torch.allclose(torch.diagonal(covmean).imag, torch.tensor(0, dtype=torch.double), atol=1e-3):
                raise ValueError(f"Imaginary component {torch.max(torch.abs(covmean.imag))} too high.")
            covmean = covmean.real

        tr_covmean = torch.trace(covmean)
        return diff.dot(diff) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * tr_covmean

    def _sqrtm(self, input_data: torch.Tensor) -> torch.Tensor:
        """Compute the square root of a matrix."""
        scipy_res, _ = linalg.sqrtm(input_data.detach().cpu().numpy().astype(np.float_), disp=False)
        return torch.from_numpy(scipy_res)

    def _cov(self, input_data: torch.Tensor, rowvar: bool = True) -> torch.Tensor:
        """
        Estimate a covariance matrix of the variables.

        Args:
            input_data: A 1-D or 2-D array containing multiple variables and observations. Each row of `m` represents a variable,
                and each column a single observation of all those variables.
            rowvar: If rowvar is True (default), then each row represents a variable, with observations in the columns.
                Otherwise, the relationship is transposed: each column represents a variable, while the rows contain
                observations.
        """
        if input_data.dim() < 2:
            input_data = input_data.view(1, -1)

        if not rowvar and input_data.size(0) != 1:
            input_data = input_data.t()

        factor = 1.0 / (input_data.size(1) - 1)
        input_data = input_data - torch.mean(input_data, dim=1, keepdim=True)
        return factor * input_data.matmul(input_data.t()).squeeze()

    def scatter(self, x, y, type='train'):

        x_min, x_max = x.min(0), x.max(0)
        X_norm = (x - x_min) / (x_max - x_min)
        # We add the labels for each digit.
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            # Position of each label.
            # plt.text(X_norm[i, 0], X_norm[i, 1], str(int(y[i].numpy())), color=plt.cm.Set1(int(y[i].numpy())), fontdict={'size': 8})
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=32, alpha=0.8, color=plt.cm.Set1(int(y[i].numpy())))

        plt.xticks([])
        plt.yticks([])
        plt.savefig('../tsne_visula_{}.png'.format(type))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=(112, 112, 16), help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images.')
    parser.add_argument('--mask-size', type=int, default=16, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.5, help='The ratio of masking area in an image (default: 0.75)')
    parser.add_argument('--dataset-path', type=str, default='/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:6", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--n_codes', type=int, default=2048)
    parser.add_argument('--n_hiddens', type=int, default=256)
    parser.add_argument('--disc_channels', type=int, default=64)
    parser.add_argument('--disc_layers', type=int, default=3, help='The default layer number is 3')
    parser.add_argument('--norm_type', type=str, default='group', choices=['batch', 'group'])
    parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])

    parser.add_argument('--train_abnorm', type=bool, default=False)
    parser.add_argument('--l1_weight', type=float, default=4.0)
    parser.add_argument('--gan_feat_weight', type=float, default=0.1)
    parser.add_argument('--image_gan_weight', type=float, default=1.0)
    parser.add_argument('--video_gan_weight', type=float, default=1.0)
    
    # setting for codebook
    parser.add_argument('--restart_thres', type=float, default=1.0)
    parser.add_argument('--no_random_restart', action='store_true')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    test_result = Test(args)
