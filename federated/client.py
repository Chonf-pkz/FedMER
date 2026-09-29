from federated.clients import client_fedavg, client_feddc, client_fedproc, client_fedprox
from federated.clients.common import build_model, evaluate_global


def _select_client(cfg):
    method = str(cfg.get("fl_method", "fedavg")).lower()
    if method == "fedprox":
        return client_fedprox
    if method == "feddc":
        return client_feddc
    if method == "fedproc":
        return client_fedproc
    return client_fedavg


def local_train(*args, **kwargs):
    cfg = kwargs.get("cfg")
    if cfg is None and len(args) >= 3:
        cfg = args[2]
    module = _select_client(cfg or {})
    return module.local_train(*args, **kwargs)
