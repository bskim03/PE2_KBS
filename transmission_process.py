from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def get_transmission(wavelength_sweep):
    transmission = {
        'vol': [],
        'l': [],
        'il': []
    }

    for ws in wavelength_sweep:
        transmission['vol'].append(ws.attrs['DCBias'] + "V")
        transmission['l'].append(np.array([float(_l) for _l in ws.find("L").text.split(",")]))
        transmission['il'].append(np.array([float(_il) for _il in ws.find("IL").text.split(",")]))

    ref = {
        'l': transmission['l'][-1],
        'il': transmission['il'][-1]
    }

    for key in transmission.keys():
        del transmission[key][-1]

    return transmission, ref


def fit_ref(ref):
    # 다항 회귀 모델 생성
    polyfit_range = range(1, 5, 1)
    models = [np.poly1d(np.polyfit(ref['l'], ref['il'], deg)) for deg in polyfit_range]
    predicted_ils = np.array([models[i](ref['l']) for i in range(len(models))])

    ref_r2_scores = [r2_score(ref['il'], predicted_ils[i]) for i in range(len(models))]
    return predicted_ils, ref_r2_scores


def process(transmission, ref):
    pass


if __name__ == '__main__':
    with open("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
        xml_data = f.read()

    soup = BeautifulSoup(xml_data, "xml")
    wavelength_sweep = soup.find_all("WavelengthSweep")
    f.close()
    trans, ref = get_transmission(wavelength_sweep)
    plt.figure(figsize=(32, 18))
    # plt.plot(trans['l'][0], trans['il'][4])
    # plt.plot(ref['l'], ref['il'])
    # plt.plot(ref['l'], trans['il'][4] - ref['il'])
    # 극대값 찾기
    peaks, _ = find_peaks(trans['il'][4])
    plt.scatter(ref['l'][peaks], trans['il'][4][peaks], s=1)
    plt.show()
