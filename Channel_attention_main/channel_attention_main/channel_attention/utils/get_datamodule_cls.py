from ..datamodules import BCICIII_IVa, BCICIII_IVaLOSO, BCICIV2a, \
    BCICIV2aLOSO, BCICIV2b, BCICIV2bLOSO, HighGamma, HighGammaLOSO, SUN_SSMVEP, Yi2014, LiuTuo


def get_datamodule_cls(dataset_name):
    if dataset_name == "bcic3":
        datamodule_cls = BCICIII_IVa
    elif dataset_name == "bcic3_loso":
        datamodule_cls = BCICIII_IVaLOSO
    elif dataset_name == "bcic2a":
        datamodule_cls = BCICIV2a
    elif dataset_name == "bcic2a_loso":
        datamodule_cls = BCICIV2aLOSO
    elif dataset_name == "bcic2b":
        datamodule_cls = BCICIV2b
    elif dataset_name == "bcic2b_loso":
        datamodule_cls = BCICIV2bLOSO
    elif dataset_name == "hgd":
        datamodule_cls = HighGamma
    elif dataset_name == "hgd_loso":
        datamodule_cls = HighGammaLOSO
    elif dataset_name == "sun_ssmvep":
        datamodule_cls = SUN_SSMVEP
    elif dataset_name == "Yi2014":
        datamodule_cls = Yi2014
    elif dataset_name == "LiuTuo":
        datamodule_cls = LiuTuo
    else:
        raise NotImplementedError(f"No dataset with name: {dataset_name}")

    return datamodule_cls
