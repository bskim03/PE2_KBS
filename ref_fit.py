import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.ticker as ticker
from lmfit import Model


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


def coeff_to_formula(coeff) -> str:
    # 다항식 계수를 받아 수식으로 반환
    result = ""
    for i in range(len(coeff)):
        result += f"{coeff[len(coeff) - i]:e}"
        if len(coeff) - i == 1:
            result += "x"
        else:
            result += f"x$^{len(coeff) - i}$"
        if coeff[len(coeff) - i - 1] >= 0:
            result += "+"
    result += f"{coeff[0]:e}"
    return result


def diode_model(voltage, isat: float, vt: float, n: float) -> float:
    # isat: saturation current
    # vt: thermal voltage
    # n: ideality factor
    return isat * (np.exp(voltage / (vt * n)) - 1)


with open("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")
WavelengthSweep = soup.find_all("WavelengthSweep")
IVMeasurement = soup.find("IVMeasurement")
f.close()

vol = np.array([float(v) for v in IVMeasurement.find("Voltage").text.split(",")])
cur = np.array([float(i) for i in IVMeasurement.find("Current").text.split(",")])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))
# ax1: Measured transmission
# ax2: Fitted transmission reference
# ax3: Measured and fitted I-V characteristics
ax4.axis('off')
fontsize = 12

# Transmission 측정 그래프 표시
for ws in WavelengthSweep[:-1]:
    vol_str = ws.attrs['DCBias'] + "V"
    l = np.array([float(_l) for _l in ws.find("L").text.split(",")])
    il = np.array([float(_il) for _il in ws.find("IL").text.split(",")])
    ax1.plot(l, il, label=vol_str)

ws = WavelengthSweep[-1]
l = np.array([float(_l) for _l in ws.find("L").text.split(",")])
il = np.array([float(_il) for _il in ws.find("IL").text.split(",")])
ax1.plot(l, il)

ax1.annotate("(-2.0V) Max transmission: -8.881 dB @ 1557.0125 nm", xy=[1530, -3], fontsize=fontsize)
ax1.annotate("(-2.0V) Min  transmission: -48.65 dB @ 1565.4627 nm", xy=[1530, -5], fontsize=fontsize)

ax1.legend(loc='lower center', ncol=3, fontsize=fontsize)
ax1.set_ylim(-52, -0)
ax1.set_title('Transmission spectra - As measured', fontsize=fontsize)
ax1.set_xlabel('Wavelength [nm]', fontsize=fontsize)
ax1.set_ylabel('Measured Transmission [dB]', fontsize=fontsize)
ax1.tick_params(axis='both', direction='in', labelsize=fontsize)

# 다항 회귀 모델 생성
polyfit_range = range(1, 7, 1)
models = [np.poly1d(np.polyfit(l, il, deg)) for deg in polyfit_range]
predicted_ils = np.array([models[i](l) for i in range(len(models))])

r2_scores = [r2_score(il, predicted_ils[i]) for i in range(len(models))]

# Ref 피팅 그래프 표시
ax2.plot(l, il, label='Reference')
for i in range(len(models)):
    ax2.plot(l, predicted_ils[i], label=f"{to_ordinal(i + 1)} fit")

ax2.annotate(coeff_to_formula(models[3]), xy=[1542, -11.5], fontsize=fontsize)
for i in range(1, len(r2_scores)):
    ax2.annotate(f"{to_ordinal(i+1)} R$^2$={r2_scores[i]}", xy=[1542, -12.25 - 0.5 * i], fontsize=fontsize)
ax2.annotate(f"Max: {max(il):.2f} dB @ {l[np.where(il==max(il))[0][0]]} nm", xy=[1528, -8], fontsize=fontsize)
ax2.annotate(f"Min: {min(il):.2f} dB @ {l[np.where(il==min(il))[0][0]]} nm", xy=[1528, -8.5], fontsize=fontsize)

ax2.set_title('Transmission spectra reference - Fitted', fontsize=fontsize)
ax2.set_xlabel('Wavelength [nm]', fontsize=fontsize)
ax2.set_ylabel('Measured Transmission [dB]', fontsize=fontsize)
ax2.tick_params(axis='both', direction='in', labelsize=fontsize)
ax2.legend(loc="upper left", fontsize=fontsize, ncol=4)
ax2.set_ylim(-16.1, -5.9)

# I-V measurement 표시
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

ax3.scatter(vol, np.abs(cur), label='Measured')
ax3.plot(x_new, np.abs(cur_pred_continuous), 'r', label=f'Fitted')

ax3.annotate(f"R$^2$={r2_score_iv}", xy=[-2.0, 1e-4], fontsize=fontsize)
ax3.annotate(f"-1V={cur[4]}A", xy=[-2.0, 1e-5], fontsize=fontsize)
ax3.annotate(f"+1V={cur[12]}A", xy=[-2.0, 1e-6], fontsize=fontsize)

ax3.set_yscale('log')
ax3.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax3.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=10))
ax3.legend(fontsize=fontsize)
ax3.set_title('I-V Characteristics', fontsize=fontsize)
ax3.set_xlabel('Voltage [V]', fontsize=fontsize)
ax3.set_ylabel('Log Absolute Current [A]', fontsize=fontsize)
ax3.tick_params(axis='both', direction='in', which='both', labelsize=fontsize)

plt.tight_layout()
plt.show()
