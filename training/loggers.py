import torch
import collections
import logging
import omegaconf
import wandb
import datetime
import glob
import os
import json

from PIL import Image


class BaseTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def stop(self):
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end) / 1000

class Timer:
    def __init__(self, info=None, log_event=None):
        self.info = info
        self.log_event = log_event

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.duration = self.start.elapsed_time(self.end) / 1000
        if self.info:
            self.info[f"duration/{self.log_event}"] = self.duration


class _StreamingMean:
    def __init__(self, val=None, counts=None):
        if val is None:
            self.mean = 0.0
            self.counts = 0
        else:
            if isinstance(val, torch.Tensor):
                val = val.data.cpu().numpy()
            self.mean = val
            if counts is not None:
                self.counts = counts
            else:
                self.counts = 1

    def update(self, mean, counts=1):
        if isinstance(mean, torch.Tensor):
            mean = mean.data.cpu().numpy()
        elif isinstance(mean, _StreamingMean):
            mean, counts = mean.mean, mean.counts * counts
        assert counts >= 0
        if counts == 0:
            return
        total = self.counts + counts
        self.mean = self.counts / total * self.mean + counts / total * mean
        self.counts = total

    def __add__(self, other):
        new = self.__class__(self.mean, self.counts)
        if isinstance(other, _StreamingMean):
            if other.counts == 0:
                return new
            else:
                new.update(other.mean, other.counts)
        else:
            new.update(other)
        return new


class StreamingMeans(collections.defaultdict):
    def __init__(self):
        super().__init__(_StreamingMean)

    def __setitem__(self, key, value):
        if isinstance(value, _StreamingMean):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, _StreamingMean(value))

    def update(self, *args, **kwargs):
        for_update = dict(*args, **kwargs)
        for k, v in for_update.items():
            self[k].update(v)

    def to_dict(self, prefix=""):
        return dict((prefix + k, v.mean) for k, v in self.items())

    def to_str(self):
        return ", ".join([f"{k} = {v:.3f}" for k, v in self.to_dict().items()])


class ConsoleLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

        self.logger.propagate = False

    @staticmethod
    def format_info(info):
        if not info:
            return str(info)
        log_groups = collections.defaultdict(dict)
        for k, v in info.to_dict().items():
            prefix, suffix = k.split("/", 1)
            log_groups[prefix][suffix] = f"{v:.3f}" if isinstance(v, float) else str(v)
        formatted_info = ""
        max_group_size = len(max(log_groups, key=len)) + 2
        max_k_size = max([len(max(g, key=len)) for g in log_groups.values()]) + 1
        max_v_size = (
            max([len(max(g.values(), key=len)) for g in log_groups.values()]) + 1
        )
        for group, group_info in log_groups.items():
            group_str = [
                f"{k:<{max_k_size}}={v:>{max_v_size}}" for k, v in group_info.items()
            ]
            max_g_size = len(max(group_str, key=len)) + 2
            group_str = "".join([f"{g:>{max_g_size}}" for g in group_str])
            formatted_info += f"\n{group + ':':<{max_group_size}}{group_str}"
        return formatted_info

    def log_iter(self, epoch_num, iter_num, num_iters, iter_info, event="epoch"):
        output_info = f"{event.upper()} {epoch_num}, ITER {iter_num}/{num_iters}:"
        output_info += self.format_info(iter_info)
        self.logger.info(output_info)

    def log_epoch(self, epoch_info, epoch_num):
        output_info = f"EPOCH {epoch_num}:"
        output_info += self.format_info(epoch_info)
        self.logger.info(output_info)


class WandbLogger:
    def __init__(self, config):
        wandb.login(key=os.environ['WANDB_KEY'].strip(), relogin=True)
        if config.train.resume_path == "":
            config_for_logger = omegaconf.OmegaConf.to_container(config)
            self.wandb_args = {
                "id": wandb.util.generate_id(),
                "project": config.exp.wandb_project,
                "name": config.exp.name,
                "config": config_for_logger,
            }
            wandb.init(**self.wandb_args, resume="allow")

            run_dir = wandb.run.dir
            print("run_dir", run_dir)

            code = wandb.Artifact("project-source", type="code")
            for path in glob.glob("**/*.py", recursive=True):
                if not path.startswith("wandb"):
                    if os.path.basename(path) != path:
                        code.add_dir(
                            os.path.dirname(path), name=os.path.dirname(path)
                        )
                    else:
                        code.add_file(os.path.basename(path), name=path)
            wandb.run.log_artifact(code)
        else:
            print(f"Resume training from {config.train.resume_path}")
            with open(config.train.resume_path, "r") as f:
                options = json.load(f)

            self.wandb_args = {
                "id": options['id'],
                "project": options['project'],
                "name": options['name'],
                "config": options['config'],
            }
            wandb.init(resume=True, **self.wandb_args)

    @staticmethod
    def log_epoch(iter_info, step):
        wandb.log(
                data={k: v.mean for k, v in iter_info.items()},
                step=step + 1,
                commit=True,
            )

    @staticmethod
    def log_special_pics(pics, captions, paths):
        to_log = {}
        for i, path in enumerate(paths):
            to_log[path] = wandb.Image(pics[i], caption=captions[path])
        wandb.log(to_log)


class BlankWandbLogger:
    def __init__(self):
        self.wandb_args = None

    def log_epoch(*args, **kwars):
        pass

    def log_special_pics(*args, **kwars):
        pass   


class TrainigLogger:
    def __init__(self, config):
        self.console_logger = ConsoleLogger("")
        if config.exp.wandb == True:
            self.wandb_logger = WandbLogger(config)
        else:
            self.wandb_logger = BlankWandbLogger()

        self.trainig_steps = config.train.steps 
        self.val_step = config.train.val_step

    def log_train_time_left(self, iter_info, step):
        float_iter_time = iter_info["duration/iter_train"].mean
        float_val_time = iter_info["duration/iter_val"].mean
        time_left = str(
            datetime.datetime.fromtimestamp(
                float_iter_time * (self.trainig_steps - step)
                + float_val_time
                * (
                    (self.trainig_steps - step) // self.val_step
                )
            )
            - datetime.datetime.fromtimestamp(0)
        )

        print()
        print(f"Step {step}/{self.trainig_steps}")
        print(f"Time left: {time_left}")
        print(f"Time per step: {iter_info['duration/iter_train'].mean :.3f}")
        print()
        print()

    def save_train_logs(self, iter_info, step):
        self.wandb_logger.log_epoch(iter_info, step)
        self.console_logger.log_epoch(iter_info, step)

        self.log_train_time_left(iter_info, step)

    def save_validation_logs(self, orig_pics, method_pics, captions, special_paths):
        log_pics = []
        for real_img, fake_img in zip(orig_pics, method_pics):
            concat_img = Image.new(
                "RGB", (real_img.width + fake_img.width, real_img.height)
            )
            concat_img.paste(real_img, (0, 0))
            concat_img.paste(fake_img, (real_img.width, 0))
            log_pics.append(concat_img)

        self.wandb_logger.log_special_pics(log_pics, captions, special_paths)
