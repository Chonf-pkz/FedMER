from federated.servers import (
    server_fedavg,
    server_fedbn,
    server_feddc,
    server_fednova,
    server_fedproc,
    server_fedprox,
)


def _select_server(cfg):
    method = str(cfg.get("fl_method", "fedavg")).lower()
    if method == "fednova":
        return server_fednova
    if method == "fedprox":
        return server_fedprox
    if method == "fedbn":
        return server_fedbn
    if method == "feddc":
        return server_feddc
    if method == "fedproc":
        return server_fedproc
    return server_fedavg


def run_stage(*args, **kwargs):
    cfg = kwargs.get("cfg")
    if cfg is None and len(args) >= 3:
        cfg = args[2]
    module = _select_server(cfg or {})
    return module.run_stage(*args, **kwargs)
