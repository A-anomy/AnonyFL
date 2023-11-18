import sys
sys.path.append("./fisco_py")
from engine import *
from configparser import ConfigParser
import fire
from common import setup_seed
import wandb

test = 0

def handle_cfg(cf:ConfigParser):
    cfg_i = cf._sections["int"]
    for key, val in cfg_i.items():
        cfg_i[key] = int(val)
    cfg_f = cf._sections["float"]
    for key, val in cfg_f.items():
        cfg_f[key] = float(val)
    cfg_b = cf._sections["bool"]
    for key, val in cfg_b.items():
        cfg_b[key] = int(val)
        if cfg_b[key] == 0:
            cfg_b[key] = False
        else:
            cfg_b[key] = True
    cfg = cf._sections["str"]
    cfg.update(cfg_f)
    cfg.update(cfg_i)
    cfg.update(cfg_b)
    return cfg
def main(config_path = "./config/fedtest.ini"):
    cf = ConfigParser()
    cf.read(config_path)
    cfg = handle_cfg(cf)
    if test == 1:
        from data_handler.cifa10_handler import Handler
        handler = Handler(cfg)
        handler.load_data()
        return
    setup_seed(cfg["np_seed"])
    if cfg["wandb"] == 1:
        wandb.init(project=cfg["project_name"], name=cfg["exp_name"], config=cfg)
    algo = eval("{}.{}".format(cfg['algorithm'],cfg['algorithm'])).run(cfg)
    if cfg["wandb"] == 1:
        wandb.finish()
    print("config file {}".format(config_path))


if __name__ == "__main__":
    fire.Fire(main)