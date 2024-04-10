from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_fedtfp_cfg(cfg):
    cfg.fedtfp = CN()
    cfg.fedtfp.mg_use = False
    cfg.fedtfp.tm_use = False
    cfg.fedtfp.gd_use = False
    cfg.fedtfp.time_window_num = 5


register_config("fedtfp", extend_fedtfp_cfg)
