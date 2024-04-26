from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def to_ordinal(n) -> str:
    # 기수를 서수로 바꿈
    if n % 10 == 1 and n % 100 != 11:
        return str(n) + "st"
    elif n % 10 == 2 and n % 100 != 12:
        return str(n) + "nd"
    elif n % 10 == 3 and n % 100 != 13:
        return str(n) + "rd"
    else:
        return str(n) + "th"


def get_data(wavelength_sweep):
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
    fit_labels = [to_ordinal(i) for i in polyfit_range]
    ref_r2_scores = [r2_score(ref['il'], predicted_ils[i]) for i in range(len(models))]
    return predicted_ils, ref_r2_scores, fit_labels


def process(transmission, ref):
    # x = transmission['l'][4]
    # y = (-8.646587371826172 + 10.310735702514648) / (1558.2638 - 1544.2563) * (x - 1558.2638) - 8.646587371826172
    processed_transmission = {
        'vol': transmission['vol'],
        'l': transmission['l'],
        'il': transmission['il'] - ref['il'],
    }
    return processed_transmission


if __name__ == '__main__':
    with open("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
        xml_data = f.read()

    soup = BeautifulSoup(xml_data, "xml")
    wavelength_sweep = soup.find_all("WavelengthSweep")
    f.close()
    trans, ref = get_data(wavelength_sweep)
    pred_ils, ref_r2_scores, fit_labels = fit_ref(ref)
    # 극대값 찾기
    peaks, _ = find_peaks(trans['il'][4])
    for i in range(len(ref['l'][peaks])):
        print(f"{trans['l'][4][peaks][i]:>9}", trans['il'][4][peaks][i])
    # plt.scatter(trans['l'][4][peaks], trans['il'][4][peaks], s=1)

    processed_trans = process(trans, ref)
    print(len(processed_trans['il']))
    plt.plot()
    for i in range(len(processed_trans['il'])):
        plt.plot(processed_trans['l'], processed_trans['il'], label="Transmission")


    plt.plot(ref['l'], ref['il'] - pred_ils[3], label="Reference")

    plt.show()
