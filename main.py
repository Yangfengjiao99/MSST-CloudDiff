import os.path
import datetime
import numpy as np
import torch
import yaml
from torch.optim import Adam
from tqdm import tqdm
import pickle
from dataset import get_dataloader
import argparse
import torch
import json
import yaml
import os
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
from model import WorkloadDiff
from sklearn.metrics import r2_score
from loss import calc_denominator,calc_quantile_CRPS
from network import prediction_fusion


def evaluate(model, test_loader, scaler=1, foldername="", config=None):
    if config is None:
        path = "./config/base.yaml"
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    feature_num = config['others']['feature_num']
    source = config.get('others', {}).get('target_predict_data', 'target_predict_data.pickle')
    mean_scaler = 0
    model.eval()
    with torch.no_grad():
        mse_total = 0
        mae_total = 0
        mape_total = 0
        rmsle_total = 0
        evalpoints_total = 0

        pre_lst = []
        tar_lst = []
        mean_r2_lst = []

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch)



                samples, c_target,c_target_2,c_target_6 ,eval_points, observed_points, observed_time = output
                pred = []
                for g in [1, 2,3]:
                    level_sample = samples[g]
                    samples_permuted = level_sample.permute(1, 0, 3, 2)
                    pred.append(samples_permuted)
                samples1 =prediction_fusion(pred)
                #truth=[c_target.permute(0,2,1),c_target_2.permute(0,2,1),c_target_6.permute(0,2,1)]

                eval_points = (eval_points).permute(0, 2, 1)

                #samples_median1, weights, crps_values=crps_based_fusion(truth,pred, eval_points,mean_scaler, scaler)
                samples_median = samples1.median(dim=1)
               # samples_median,weights, errors =error_inverse_weighted_fusion(pred,truth,eval_points)

                c_target = c_target.permute(0, 2, 1)
        
                observed_points = observed_points.permute(0, 2, 1)

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples1)

                # mse
                mse_current = (((samples_median.values - c_target) * eval_points) ** 2) * (scaler ** 2)

                # mae
                mae_current = (torch.abs((samples_median.values - c_target) * eval_points)) * scaler

                target_eval = c_target[eval_points == 1.].reshape(-1, feature_num)
                predict_eval = samples_median.values[eval_points == 1.].reshape(-1, feature_num)

                if True in np.isnan(predict_eval.cpu()):
                    print("The results contain null values, the program has stopped.")
                    exit(0)

                pre_lst.append(predict_eval)
                tar_lst.append(target_eval)
                mean_r2_lst.append(r2_score(target_eval.cpu(), predict_eval.cpu()))

                # mape
                mape_current = (
                    torch.abs((samples_median.values - c_target) / c_target * eval_points)
                )

                mape_current = torch.where(torch.isnan(mape_current), torch.full_like(mape_current, 0), mape_current)
                mape_current = torch.where(torch.isinf(mape_current), torch.full_like(mape_current, 0), mape_current)

                # rmsle
                rmsle_current = ((torch.log(c_target * eval_points + 1) - torch.log(samples_median.values * eval_points + 1)) ** 2 * eval_points)

                # Record all prediction results.
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mape_total += mape_current.sum().item()
                rmsle_total += rmsle_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "mape_total": mape_total / evalpoints_total,
                        "rmsle_total": np.sqrt(rmsle_total / evalpoints_total),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            if evalpoints_total == 0:
              print("warning")
              return
            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_observed_point = torch.cat(all_observed_point, dim=0)
            all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)
            if foldername:
                with open(os.path.join(foldername, f"generated_outputs_nsample{3}.pk"), "wb") as f:
                    pickle.dump(
                        [
                            all_generated_samples.cpu(),
                            all_target.cpu(),
                            all_evalpoint.cpu(),
                            all_observed_point.cpu(),
                            all_observed_time.cpu(),
                            scaler,
                            mean_scaler,
                        ],
                        f,
                    )

            # crps
            CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

            y_target = torch.cat(tar_lst, dim=0).cpu()
            y_predict = torch.cat(pre_lst, dim=0).cpu()
            r_2 = r2_score(y_target, y_predict)

            with open(
                    foldername + "/result_nsample" + str(0) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        mape_total / evalpoints_total,
                        r_2,
                        CRPS,
                        np.sqrt(rmsle_total / evalpoints_total)
                    ],
                    f,
                )

            lst = [('key', 'value'), ('RMSE', np.sqrt(mse_total / evalpoints_total)),
                   ('MAE', mae_total / evalpoints_total),
                   ('MAPE', mape_total / evalpoints_total), ('R_2', r_2), ('CRPS', CRPS),
                   ('RMSLE', np.sqrt(rmsle_total / evalpoints_total)),
                   ('epoch', config['train']['epochs']), ('dir', config['others']['dir_dataset']),
                   ('model_folder', config['others']['model_folder'])]

            for i, j in lst:
                print('{:^15}{:^10}'.format(i, j))

            # Save the results as a file.
            with open(os.path.join(foldername, source), 'wb') as f:
                pickle.dump({'target': torch.cat(tar_lst, dim=0).cpu(), 'predict': torch.cat(pre_lst, dim=0).cpu()}, f)




def train(model, config, train_loader,test_loader, foldername="", ):

    optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"
    else:
        output_path = './'

    p1 = int(0.4 * config["train"]["epochs"])
    p2 = int(0.8 * config["train"]["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    train_loss_lst = []
    all_epoch_tarin_loss_lst = []

    for epoch_no in range(config["train"]["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss,likelihoods = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                    "avg_epoch_loss": avg_loss / batch_no,
                    "epoch": epoch_no,
                    },
                    refresh=False,
                )
                train_loss_lst.append(loss.item())
            lr_scheduler.step()

        all_epoch_tarin_loss_lst.append(train_loss_lst)
    torch.save(model.state_dict(), output_path)

    with open(os.path.join(foldername, 'train_val_loss.pk'), 'wb') as f:
        pickle.dump({
              'train_loss': all_epoch_tarin_loss_lst.append}, f)

    print("finish training 。")


def single_process(config, foldername,mg_dict,share_ratio_list, args):
    device = torch.device(args.device)
    model = WorkloadDiff(config,share_ratio_list,device).to(device)
    batch_size = config["train"]["batch_size"]
    train_loader, valid_loader, test_loader = get_dataloader(
        mg_dict=mg_dict,
        seed=args.seed,
        batch_size=batch_size,
        config=config
    )
    if args.modelfolder == "":

        train(
            model,
            config,
            train_loader,
            test_loader,
            foldername=foldername,
        )
    # Model loading
        evaluate(model, test_loader, scaler=1, foldername=foldername, config=config)
    else:
        head, tail = os.path.split(foldername)
        model_path = os.path.join(head, args.modelfolder + "/model.pth")
        model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
        print(f"Loaded model from: {model_path}")
        evaluate(model, test_loader, scaler=1, foldername=foldername, config=config)
if __name__ == '__main__':

    torch.set_printoptions(profile="full")
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument(
        "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
    )
    #parser.add_argument("--modelfolder", type=str, default="./WorkloadDiff_20251112_085724")
    #parser.add_argument("--modelfolder", type=str, default="./WorkloadDiff_20251220_154052")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument('--mg_dict', type=str, default='1_2',
                        help='the multi-granularity list, 1_4 means 1h and 4h, 1_4_8 means 1h, 4h and 8h ')
    parser.add_argument('--share_ratio_list', type=str, default="1_0.6",
                        help='the share ratio list, 1_0.9, means that for the second granularity, 90% of the diffusion steps are shared with the finest granularity.')
    parser.add_argument("--nsample", type=int, default=10)
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:23500')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    args = parser.parse_args()
    start = datetime.datetime.now()
    print('start: ', start)
    mg_dict = {'2': 2, '4': 4,'6': 6}
    print(f"mg_dict:{mg_dict}")

    share_ratio_list = [float(i) for i in str(args.share_ratio_list).split('_')]
    print(f"share_ratio_list:{share_ratio_list}")
    # config
    path = "./config/base.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        print(config)
    config["model"]["test_missing_ratio"] = args.testmissingratio

    # model folder
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_logs = config["others"]["dir_dataset"]

    # Model storage folder
    tmp = dir_logs.split("/")[:-3]
    foldername = os.path.join('/'.join(tmp), "logs/WorkloadDiff_" + current_time)
    config["others"]["model_folder"] = foldername

    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(json.dumps(config, indent=4))
    single_process(config, foldername,mg_dict,share_ratio_list, args)
    end = datetime.datetime.now()
    print('end: ', end)
    print('TIME USAGE: ', end - start)