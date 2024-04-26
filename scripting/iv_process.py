from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics import r2_score
from lmfit import Model


def diode_model(voltage, isat: float, vt: float, n: float) -> float:
    # isat: saturation current
    # vt: thermal voltage
    # n: ideality factor
    return isat * (np.exp(voltage / (vt * n)) - 1)


def get_data(ivmeasurement):
    vol = np.array([float(v) for v in ivmeasurement.find("Voltage").text.split(",")])
    cur = np.array([float(i) for i in ivmeasurement.find("Current").text.split(",")])
    return vol, cur


def fit(vol, cur):
    model_pos = Model(diode_model)
    params = model_pos.make_params(isat=5e-6, vt=26e-3, n=1.0)
    params['n'].min = 1
    result_pos = model_pos.fit(cur[8:], voltage=vol[8:], params=params)

    model_neg = Model(diode_model)
    params = model_neg.make_params(isat=1e-1, vt=26e-3, n=1.0)
    params['n'].min = 1
    result_neg = model_neg.fit(cur[:8], voltage=vol[:8], params=params)

    params_new_pos = result_pos.params.values()
    params_new_neg = result_neg.params.values()

    x_new = np.linspace(-2, 1, 1000)
    cur_pred_continuous_pos = diode_model(x_new_pos := x_new[667:], *params_new_pos)
    cur_pred_continuous_neg = diode_model(x_new_neg := x_new[:667], *params_new_neg)
    cur_pred_continuous = np.concatenate((cur_pred_continuous_neg, cur_pred_continuous_pos))

    cur_pred_discrete_pos = diode_model(vol[8:], *params_new_pos)
    cur_pred_discrete_neg = diode_model(vol[:8], *params_new_neg)
    cur_pred_discrete = np.concatenate((cur_pred_discrete_neg, cur_pred_discrete_pos))

    r2_score_iv = r2_score(cur, cur_pred_discrete)

    return x_new, cur_pred_continuous, r2_score_iv
