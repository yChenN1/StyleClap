import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from clap_module import LPLoss, LPMetrics, lp_gather_features
from clap_module.utils import do_mixup, get_mix_lambda
from .distributed import is_master
from .zero_shot import zero_shot_eval


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def train_one_epoch(
        model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None, extra_suffix=""
):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    model.train()
    loss = LPLoss(args.lp_loss)

    dataloader, sampler = data["train"].dataloader, data["train"].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # for toy dataset
    # if args.dataset_type == "toy":
    #     dataloader.dataset.generate_queue()

    loss_gender = AverageMeter()
    loss_age = AverageMeter()
    loss_speed = AverageMeter()
    loss_pitch = AverageMeter()
    loss_energy = AverageMeter()
    loss_emotion = AverageMeter()
    loss_total = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        global_step = num_batches_per_epoch * epoch + i

        if isinstance(scheduler, dict):
            for s in scheduler.values():
                s(global_step)
        else:
            scheduler(global_step)

        audio = batch # contains mel_spec, wavform, and longer list
        class_label = batch['class_label']
        # audio = audio.to(device=device, non_blocking=True)
        class_label = class_label.to(device=device, non_blocking=True)

        if args.mixup:
            # https://github.com/RetroCirce/HTS-Audio-Transformer/blob/main/utils.py#L146
            mix_lambda = torch.from_numpy(get_mix_lambda(0.5, len(audio["waveform"]))).to(device)
            class_label = do_mixup(class_label, mix_lambda)
        else:
            mix_lambda = None

        data_time_m.update(time.time() - end)
        if isinstance(optimizer, dict):
            for o_ in optimizer.values():
                o_.zero_grad()
        else:
            optimizer.zero_grad()

        with autocast():
            pred = model(audio, mix_lambda=mix_lambda, device=device)
            # gender_loss = loss(pred[:, :2], class_label[:, 0])
            # age_loss = loss(pred[:, 2:7], class_label[:, 1])
            # speed_loss = loss(pred[:, 7:10], class_label[:, 2])
            # pitch_loss = loss(pred[:, 10:13], class_label[:, 3])
            # energy_loss = loss(pred[:, 13:16], class_label[:, 4])
            # emotion_loss = loss(pred[:, 16:], class_label[:, 5])
            style_loss = loss(pred, class_label)

            # total_loss = (gender_loss*0.0 + age_loss*0.1 + speed_loss*1.5 + pitch_loss*1.0 + energy_loss*0.5 + emotion_loss*3) / 6
            # total_loss = (gender_loss*0.0 + age_loss*0.0 + speed_loss*0.0 + pitch_loss*0.0 + energy_loss*0.0 + emotion_loss*6) / 6
            total_loss = style_loss 

        if isinstance(optimizer, dict):
            if scaler is not None:
                scaler.scale(total_loss).backward()
                for o_ in optimizer.values():
                    if args.horovod:
                        o_.synchronize()
                        scaler.unscale_(o_)
                        with o_.skip_synchronize():
                            scaler.step(o_)
                    else:
                        scaler.step(o_)
                scaler.update()
            else:
                total_loss.backward()
                for o_ in optimizer.values():
                    o_.step()
        else:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).clap_model.logit_scale_a.clamp_(0, math.log(100))
            unwrap_model(model).clap_model.logit_scale_t.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        if is_master(args) and (i % 10 == 0 or batch_count == num_batches_per_epoch):
            if isinstance(audio, dict):
                batch_size = len(audio["waveform"])
            else:
                batch_size = len(audio)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            iterations_per_epoch = samples_per_epoch // (batch_size * args.world_size) + 1

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_total.update(total_loss.item(), batch_size)
            # loss_gender.update(gender_loss.item(), batch_size)
            # loss_age.update(age_loss.item(), batch_size)
            # loss_speed.update(speed_loss.item(), batch_size)
            # loss_pitch.update(pitch_loss.item(), batch_size)
            # loss_energy.update(energy_loss.item(), batch_size)
            # loss_emotion.update(emotion_loss.item(), batch_size)
            if isinstance(optimizer, dict):
                logging.info(
                    f"Train Epoch: {epoch} [{batch_count:>{sample_digits}}/{iterations_per_epoch} ({percent_complete:.0f}%)] "
                    # f"Gender Loss: {loss_gender.val:#.5g} ({loss_gender.avg:#.4g}) "
                    # f"Age Loss: {loss_age.val:#.5g} ({loss_age.avg:#.4g}) "
                    # f"Speed Loss: {loss_speed.val:#.5g} ({loss_speed.avg:#.4g}) "
                    # f"Pitch Loss: {loss_pitch.val:#.5g} ({loss_pitch.avg:#.4g}) "
                    # f"Energy Loss: {loss_energy.val:#.5g} ({loss_energy.avg:#.4g}) "
                    # f"Emotion Loss: {loss_emotion.val:#.5g} ({loss_emotion.avg:#.4g}) "
                    f"Total Loss: {loss_total.val:#.5g} ({loss_total.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f} "
                    f"LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]}"
                )
                log_data = {
                    "loss": loss_total.val,
                    # "gender_loss": loss_gender.val,
                    # "age_loss": loss_age.val,
                    # "speed_loss": loss_speed.val,
                    # "pitch_loss": loss_pitch.val,
                    # "energy_loss": loss_energy.val,
                    # "emotion_loss": loss_emotion.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "lr": [o_.param_groups[0]["lr"] for o_ in optimizer.values()],
                }
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{batch_count:>{sample_digits}}/{iterations_per_epoch} ({percent_complete:.0f}%)] "
                    # f"Gender Loss: {loss_gender.val:#.5g} ({loss_gender.avg:#.4g}) "
                    # f"Age Loss: {loss_age.val:#.5g} ({loss_age.avg:#.4g}) "
                    # f"Speed Loss: {loss_speed.val:#.5g} ({loss_speed.avg:#.4g}) "
                    # f"Pitch Loss: {loss_pitch.val:#.5g} ({loss_pitch.avg:#.4g}) "
                    # f"Energy Loss: {loss_energy.val:#.5g} ({loss_energy.avg:#.4g}) "
                    # f"Emotion Loss: {loss_emotion.val:#.5g} ({loss_emotion.avg:#.4g}) "
                    f"Total Loss: {loss_total.val:#.5g} ({loss_total.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f} "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "loss": loss_total.val,
                    # "gender_loss": loss_gender.val,
                    # "age_loss": loss_age.val,
                    # "speed_loss": loss_speed.val,
                    # "pitch_loss": loss_pitch.val,
                    # "energy_loss": loss_energy.val,
                    # "emotion_loss": loss_emotion.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            for name, val in log_data.items():
                name = f"train{extra_suffix}/{name}"
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, global_step)
                if args.wandb:
                    assert wandb is not None, "Please install wandb."
                    wandb.log({name: val, "step": global_step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def evaluate(model, data, epoch, args, tb_writer=None, extra_suffix=""):
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    # 所有进程都需要初始化评估工具
    metric_names = args.lp_metrics.split(',')
    eval_tool = LPMetrics(metric_names=metric_names) if metric_names[0] != '' else None
    
    if is_master(args):
        print('Evaluating...')

    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    
    if "val" in data and (
            args.val_frequency
            and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        # 所有进程都需要获取dataloader
        if args.parallel_eval:
            dataloader, sampler = data["val"].dataloader, data["val"].sampler
            if args.distributed and sampler is not None:
                sampler.set_epoch(epoch)
            samples_per_val = dataloader.num_samples
        else:
            dataloader = data["val"].dataloader
            samples_per_val = dataloader.num_samples
        
        # 所有进程都初始化评估信息字典（在GPU上）
        eval_info = {
            'pred': {'style': []},
            'target': {'style': []}
        }
        
        num_samples = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                audio = batch
                class_label = batch['class_label']
                class_label = class_label.to(device=device, non_blocking=True)

                with autocast():
                    pred = model(audio, device=device)
                    
                    # 如果是并行评估，收集所有进程的特征
                    if args.parallel_eval:
                        pred, class_label = lp_gather_features(
                            pred, class_label, args.world_size, args.horovod
                        )

                    # 存储预测值和目标值（保持在GPU上）
                    eval_info['pred']['style'].append(pred.detach())
                    eval_info['target']['style'].append(class_label.detach())

                num_samples += class_label.shape[0]

                if is_master(args) and (i % 10) == 0:
                    logging.info(f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]")
        
        # 所有进程都拼接结果（保持在GPU上）
        for attr in eval_info['pred']:
            if eval_info['pred'][attr]:  # 确保列表不为空
                eval_info['pred'][attr] = torch.cat(eval_info['pred'][attr], 0)
                eval_info['target'][attr] = torch.cat(eval_info['target'][attr], 0)
            else:
                # 如果没有数据，创建空的tensor（在GPU上）
                eval_info['pred'][attr] = torch.tensor([], device=device)
                eval_info['target'][attr] = torch.tensor([], device=device)

        # 使用分布式通信收集所有进程的结果
        if args.distributed and not args.horovod:
            # 确保所有进程都有相同形状的张量用于all_gather
            if eval_info['pred']['style'].shape[0] == 0:
                # 如果没有数据，创建一个占位符张量
                placeholder_pred = torch.zeros(1, eval_info['pred']['style'].shape[1] if len(eval_info['pred']['style'].shape) > 1 else 1, device=device)
                placeholder_target = torch.zeros(1, eval_info['target']['style'].shape[1] if len(eval_info['target']['style'].shape) > 1 else 1, device=device)
            else:
                placeholder_pred = eval_info['pred']['style']
                placeholder_target = eval_info['target']['style']
            
            # 收集所有进程的预测结果
            pred_list = [torch.zeros_like(placeholder_pred) for _ in range(args.world_size)]
            target_list = [torch.zeros_like(placeholder_target) for _ in range(args.world_size)]
            
            torch.distributed.all_gather(pred_list, placeholder_pred)
            torch.distributed.all_gather(target_list, placeholder_target)
            
            # 主进程拼接所有结果
            if is_master(args):
                # 过滤掉空的占位符张量
                valid_preds = [pred for pred in pred_list if pred.numel() > 0 and not torch.all(pred == 0)]
                valid_targets = [target for target in target_list if target.numel() > 0 and not torch.all(target == 0)]
                
                if valid_preds and valid_targets:
                    all_preds = torch.cat(valid_preds, 0).cpu()
                    all_targets = torch.cat(valid_targets, 0).cpu()
                else:
                    all_preds = torch.tensor([])
                    all_targets = torch.tensor([])
            else:
                all_preds = None
                all_targets = None
        else:
            all_preds = eval_info['pred']['style'].cpu()
            all_targets = eval_info['target']['style'].cpu()

        # 只有主进程计算最终指标
        if is_master(args) and eval_tool is not None:
            if all_preds.numel() > 0 and all_targets.numel() > 0:  # 确保有数据
                attr_metrics = eval_tool.evaluate_mertics(all_preds, all_targets)
                metric_dict = {}
                for metric_name, value in attr_metrics.items():
                    metric_dict[f"style_{metric_name}"] = value
                
                metrics.update(metric_dict)
                metrics.update({"epoch": epoch})

                # 日志输出
                log_lines = [f"Eval Epoch: {epoch}"]
                attr_metrics_lines = [f"{m}: {round(metrics[m], 4):.4f}" 
                                    for m in metrics if m.startswith("style_")]
                if attr_metrics_lines:
                    log_lines.append("  style:")
                    log_lines.extend([f"    {m}" for m in attr_metrics_lines])
                
                logging.info("\n".join(log_lines))
                
                if args.save_logs:
                    for name, val in metrics.items():
                        if tb_writer is not None:
                            tb_writer.add_scalar(f"val{extra_suffix}/{name}", val, epoch)

                    with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                        f.write(json.dumps(metrics))
                        f.write("\n")

                if args.wandb:
                    assert wandb is not None, "Please install wandb."
                    log_data = {f"val{extra_suffix}/{name}": val for name, val in metrics.items()}
                    log_data["epoch"] = epoch
                    wandb.log(log_data)
    
    # 所有进程都需要同步，确保验证完成后再继续训练
    if args.distributed:
        torch.distributed.barrier()
    
    return metrics

def evaluate2(model, data, epoch, args, tb_writer=None, extra_suffix=""):
    metrics = {}
    if not args.parallel_eval:
        if not is_master(args):
            torch.distributed.barrier()
            return metrics
    device = torch.device(args.device)
    model.eval()

    # CHANGE
    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    # metrics.update(zero_shot_metrics)
    if is_master(args):
        print('Evaluating...')
        metric_names = args.lp_metrics.split(',')
        eval_tool = LPMetrics(metric_names=metric_names)

    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    if "val" in data and (
            args.val_frequency
            and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        if args.parallel_eval:
            dataloader, sampler = data["val"].dataloader, data["val"].sampler
            if args.distributed and sampler is not None:
                sampler.set_epoch(epoch)
            samples_per_val = dataloader.num_samples
        else:
            dataloader = data["val"].dataloader
            num_samples = 0
            samples_per_val = dataloader.num_samples
            
        # 初始化评估信息字典，为每个属性创建列表
        eval_info = {
            'pred': {
                'style': []
                # 'gender': [],
                # 'age': [],
                # 'speed': [],
                # 'pitch': [],
                # 'energy': [],
                # 'emotion': []
            },
            'target': {
                'style': []
                # 'gender': [],
                # 'age': [],
                # 'speed': [],
                # 'pitch': [],
                # 'energy': [],
                # 'emotion': []
            }
        }
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                audio = batch  # 包含mel_spec、wavform等
                class_label = batch['class_label']
                        
                class_label = class_label.to(device=device, non_blocking=True)

                with autocast():
                    pred = model(audio, device=device)
                    if args.parallel_eval:
                        pred, class_label = lp_gather_features(
                            pred, class_label, args.world_size, args.horovod
                        )

                    # 按属性存储预测值和目标值
                    eval_info['pred']['style'].append(pred)
                    eval_info['target']['style'].append(class_label)

                    # eval_info['pred']['gender'].append(pred[:, :2])
                    # eval_info['pred']['age'].append(pred[:, 2:7])
                    # eval_info['pred']['speed'].append(pred[:, 7:10])
                    # eval_info['pred']['pitch'].append(pred[:, 10:13])
                    # eval_info['pred']['energy'].append(pred[:, 13:16])
                    # eval_info['pred']['emotion'].append(pred[:, 16:])
                    
                    # eval_info['target']['gender'].append(class_label[:, 0])
                    # eval_info['target']['age'].append(class_label[:, 1])
                    # eval_info['target']['speed'].append(class_label[:, 2])
                    # eval_info['target']['pitch'].append(class_label[:, 3])
                    # eval_info['target']['energy'].append(class_label[:, 4])
                    # eval_info['target']['emotion'].append(class_label[:, 5:])
                    
                num_samples += class_label.shape[0]

                if (i % 10) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]"
                    )
                    
            if is_master(args):
                # 拼接所有批次的结果
                for attr in eval_info['pred']:
                    eval_info['pred'][attr] = torch.cat(eval_info['pred'][attr], 0).cpu()
                    eval_info['target'][attr] = torch.cat(eval_info['target'][attr], 0).cpu()
                
                # 为每个属性计算评估指标，避免覆盖
                metric_dict = {}
                # attributes = ['gender', 'age', 'speed', 'pitch', 'energy', 'emotion']
                attributes = ['style']
                for attr in attributes:
                    # 假设evaluate_mertics返回一个字典，包含该属性的各项指标
                    attr_metrics = eval_tool.evaluate_mertics(
                        eval_info['pred'][attr], 
                        eval_info['target'][attr]
                    )
                    # 为每个指标添加属性前缀，避免命名冲突
                    for metric_name, value in attr_metrics.items():
                        metric_dict[f"{attr}_{metric_name}"] = value

                metrics.update(metric_dict)
                if "epoch" not in metrics.keys():
                    metrics.update({"epoch": epoch})

    if is_master(args):
        if not metrics:
            return metrics

        # 格式化日志输出，按属性分组展示
        log_lines = [f"Eval Epoch: {epoch}"]
        attributes = ['style']
        # attributes = ['gender', 'age', 'speed', 'pitch', 'energy', 'emotion']
        for attr in attributes:
            attr_metrics = [f"{m}: {round(metrics[m], 4):.4f}" 
                          for m in metrics if m.startswith(f"{attr}_")]
            if attr_metrics:
                log_lines.append(f"  {attr}:")
                log_lines.extend([f"    {m}" for m in attr_metrics])
        
        logging.info("\n".join(log_lines))
        
        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val{extra_suffix}/{name}", val, epoch)

            with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        if args.wandb:
            assert wandb is not None, "Please install wandb."
            log_data = {f"val{extra_suffix}/{name}": val for name, val in metrics.items()}
            log_data["epoch"] = epoch
            wandb.log(log_data)
        # 最关键的一步：所有进程必须在这里同步，等待主进程验证完成
        torch.distributed.barrier()
        return metrics
    else:
        return metrics



    # if "val" in data and (
    #         args.val_frequency
    #         and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    # ):
    #     if args.parallel_eval:
    #         dataloader, sampler = data["val"].dataloader, data["val"].sampler
    #         if args.distributed and sampler is not None:
    #             sampler.set_epoch(epoch)
    #         samples_per_val = dataloader.num_samples
    #     else:
    #         dataloader = data["val"].dataloader
    #         num_samples = 0
    #         samples_per_val = dataloader.num_samples
            
    #     eval_info = {
    #         'pred': [],
    #         'target': []
    #     }
    #     with torch.no_grad():
    #         for i, batch in enumerate(dataloader):
    #             audio = batch # contains mel_spec, wavform, and longer list
    #             __import__("ipdb").set_trace()
    #             class_label = batch['class_label']
                           
    #             # audio = audio.to(device=device, non_blocking=True)
    #             class_label = class_label.to(device=device, non_blocking=True)

    #             with autocast():
    #                 pred = model(audio, device=device)
    #                 if args.parallel_eval:
    #                     pred, class_label = lp_gather_features(pred, class_label, args.world_size, args.horovod)

    #                 eval_info['pred']['gender'].append(pred[:, :2])
    #                 eval_info['pred']['age'].append(pred[:, 2:7])
    #                 eval_info['pred']['speed'].append(pred[:, 7:10])
    #                 eval_info['pred']['pitch'].append(pred[:, 10:13])
    #                 eval_info['pred']['energy'].append(pred[:, 13:16])
    #                 eval_info['pred']['emotion'].append(pred[:, 16:])
    #                 eval_info['target']['gender'].append(class_label[:, 0])
    #                 eval_info['target']['age'].append(class_label[:, 1])
    #                 eval_info['target']['speed'].append(class_label[:, 2])
    #                 eval_info['target']['pitch'].append(class_label[:, 3])
    #                 eval_info['target']['energy'].append(class_label[:, 4])
    #                 eval_info['target']['emotion'].append(class_label[:, 5:])
                    
    #             num_samples += class_label.shape[0]

    #             if (i % 100) == 0:  # and i != 0:
    #                 logging.info(
    #                     f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]"
    #                 )
                    
    #         if is_master(args):
    #             eval_info['pred'] = torch.cat(eval_info['pred'], 0).cpu()
    #             eval_info['target'] = torch.cat(eval_info['target'], 0).cpu()
    #             metric_dict = eval_tool.evaluate_mertics(eval_info['pred']['gender'], eval_info['target']['gender'])
    #             metric_dict = eval_tool.evaluate_mertics(eval_info['pred']['age'], eval_info['target']['age'])
    #             metric_dict = eval_tool.evaluate_mertics(eval_info['pred']['speed'], eval_info['target']['speed'])
    #             metric_dict = eval_tool.evaluate_mertics(eval_info['pred']['pitch'], eval_info['target']['pitch'])
    #             metric_dict = eval_tool.evaluate_mertics(eval_info['pred']['energy'], eval_info['target']['energy'])
    #             metric_dict = eval_tool.evaluate_mertics(eval_info['pred']['emotion'], eval_info['target']['emotion'])

    #             metrics.update(metric_dict)
    #             if "epoch" not in metrics.keys():
    #                 metrics.update({"epoch": epoch})

    # if is_master(args):
    #     if not metrics:
    #         return metrics

    #     logging.info(
    #         f"Eval Epoch: {epoch} "
    #         + "\n".join(
    #             [
    #                 "\t".join([f"{m}: {round(metrics[m], 4):.4f}" ])
    #                 for m in metrics
    #             ]
    #         )
    #     )
    #     if args.save_logs:
    #         for name, val in metrics.items():
    #             if tb_writer is not None:
    #                 tb_writer.add_scalar(f"val{extra_suffix}/{name}", val, epoch)

    #         with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
    #             f.write(json.dumps(metrics))
    #             f.write("\n")

    #     if args.wandb:
    #         assert wandb is not None, "Please install wandb."
    #         for name, val in metrics.items():
    #             wandb.log({f"val{extra_suffix}/{name}": val, "epoch": epoch})

    #     return metrics
    # else:
    #     return metrics
