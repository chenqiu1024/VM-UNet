import os
import shutil
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import per_sample_dice, save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    mean_loss = float(np.mean(loss_list))
    log_info = f'train epoch: {epoch}, loss: {mean_loss:.4f}'
    print(log_info)
    logger.info(log_info)
    writer.add_scalar('train_loss_epoch', mean_loss, epoch)
    return step, mean_loss


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return float(np.mean(loss_list))


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    dice_records = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            dice = per_sample_dice(msk, out, config.threshold)
            sample_name = None
            if hasattr(test_loader.dataset, 'data'):
                sample_name = os.path.basename(test_loader.dataset.data[i][0])
            saved_path = None
            if i % config.save_interval == 0:
                _, saved_path = save_imgs(
                    img, msk, out, i, config.work_dir + 'outputs/', config.datasets,
                    config.threshold, test_data_name=test_data_name,
                    sample_name=sample_name, dice=dice,
                )
            dice_records.append({
                'index': i,
                'dice': dice,
                'name': sample_name if sample_name is not None else str(i),
                'path': saved_path,
            })

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)
        _export_dice_extremes(dice_records, config.work_dir + 'outputs/', logger)

    return np.mean(loss_list)


def _export_dice_extremes(records, out_dir, logger, k=10):
    if not records:
        return
    ranked = sorted(records, key=lambda r: (-r['dice'], r['index']))
    lowest = sorted(records, key=lambda r: (r['dice'], r['index']))[:k]
    highest = ranked[:k]
    summary_path = os.path.join(out_dir, 'dice_per_sample.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('index\tdice\tname\tpath\n')
        for r in ranked:
            f.write(f"{r['index']}\t{r['dice']:.6f}\t{r['name']}\t{r['path']}\n")

    def copy_group(group, folder, reverse_rank):
        dest_dir = os.path.join(out_dir, folder)
        os.makedirs(dest_dir, exist_ok=True)
        lines = []
        for rank, r in enumerate(group, start=1):
            if not r['path'] or not os.path.exists(r['path']):
                lines.append(f"{rank}\t{r['dice']:.6f}\t{r['name']}\tMISSING")
                continue
            stem = os.path.splitext(r['name'])[0]
            dest_name = f"{rank:02d}_dice{r['dice']:.4f}_{r['index']:04d}_{stem}.png"
            dest = os.path.join(dest_dir, dest_name)
            shutil.copy2(r['path'], dest)
            lines.append(f"{rank}\t{r['dice']:.6f}\t{r['name']}\t{dest}")
        note = 'highest' if reverse_rank else 'lowest'
        log_info = f'{note} {len(group)} dice samples saved to {dest_dir}'
        print(log_info)
        logger.info(log_info)
        for line in lines:
            print(line)
            logger.info(line)

    copy_group(highest, 'dice_highest', True)
    copy_group(lowest, 'dice_lowest', False)