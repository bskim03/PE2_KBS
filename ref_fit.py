import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics import r2_score


def to_ordinal(n):
    if n % 10 == 1 and n % 100 != 11:
        return str(n) + "st"
    elif n % 10 == 2 and n % 100 != 12:
        return str(n) + "nd"
    elif n % 10 == 3 and n % 100 != 13:
        return str(n) + "rd"
    else:
        return str(n) + "th"


def coeff_to_formula(coeff):
    result = ""
    for i in range(len(coeff)):
        result += f"{coeff[len(coeff) - i]:.1e}"
        if len(coeff) - i == 1:
            result += "x"
        else:
            result += f"x$^{len(coeff) - i}$"
        if coeff[len(coeff) - i - 1] >= 0:
            result += "+"
    result += f"{coeff[0]:.1e}"
    return result


with open("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")

WavelengthSweep = soup.find_all("WavelengthSweep")

ref = WavelengthSweep[-1]
l = np.array([float(l) for l in ref.find("L").text.split(",")])
il = np.array([float(il) for il in ref.find("IL").text.split(",")])

polyfit_range = range(1, 5, 1)
# 다항 회귀 모델 생성
models = [np.poly1d(np.polyfit(l, il, deg)) for deg in polyfit_range]
print(models[1])
print(models[1].coeffs)
pred_ils = np.array([models[i](l) for i in range(len(models))])
r2_scores = [r2_score(il, pred_ils[i]) for i in range(len(models))]
print(r2_scores)
print(r2_scores.index(max(r2_scores)))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax4.axis('off')
fontsize = 12

# Transmission 측정 그래프 표시
for ws in WavelengthSweep[:-1]:
    vol: str = ws.attrs['DCBias'] + "V"
    l = np.array([float(_l) for _l in ws.find("L").text.split(",")])
    il = np.array([float(_il) for _il in ws.find("IL").text.split(",")])
    ax1.plot(l, il, label=vol)

ws = WavelengthSweep[-1]
l = np.array([float(_l) for _l in ws.find("L").text.split(",")])
il = np.array([float(_il) for _il in ws.find("IL").text.split(",")])
ax1.plot(l, il)
ax1.legend(loc='lower center', ncol=3, fontsize=fontsize)
ax1.set_title('Transmission spectra - As measured', fontsize=fontsize)
ax1.set_xlabel('Wavelength [nm]', fontsize=fontsize)
ax1.set_ylabel('Measured Transmission [dB]', fontsize=fontsize)
ax1.tick_params(axis='both', labelsize=fontsize)


# Ref 피팅 그래프 표시
ax2.plot(l, il)
for i in range(len(models)):
    ax2.plot(l, pred_ils[i], label=to_ordinal(i + 1))
ax2.set_title('Transmission spectra - Processed and fitted', fontsize=fontsize)
ax2.set_xlabel('Wavelength [nm]', fontsize=fontsize)
ax2.set_ylabel('Measured Transmission [dB]', fontsize=fontsize)
ax2.tick_params(axis='both', labelsize=fontsize)

for i in range(3):
    ax2.annotate(coeff_to_formula(models[i+1]), xy=[1545, -11.5-1.5*i], fontsize=fontsize)
    ax2.annotate(f"R$^2$={r2_scores[i+1]}", xy=[1545, -12-1.5*i], fontsize=fontsize)
ax2.legend(loc="upper left", fontsize=fontsize, ncol=3)
plt.tight_layout()
plt.show()
