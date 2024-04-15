from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from lmfit import Model

with open("..\HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")

IVMeasurement = soup.find("IVMeasurement")
vol = np.array([float(v) for v in IVMeasurement.find("Voltage").text.split(",")])
cur = np.array([float(c) for c in IVMeasurement.find("Current").text.split(",")])


def diode_model(voltage, isat, vt, n):
    # isat: saturation current
    # vt: thermal voltage
    # n: ideality factor
    return isat * (np.exp(voltage / (vt * n)) - 1)


model_pos = Model(diode_model)
params = model_pos.make_params(isat=1e-5, vt=26e-3, n=1)
params['n'].min = 1
result_pos = model_pos.fit(cur[8:], voltage=vol[8:], params=params)
# print(result.fit_report())
# print(result.params)
model_neg = Model(diode_model)
params = model_neg.make_params(isat=0, vt=26e-3, n=1)
params['n'].min = 1
result_neg = model_neg.fit(cur[:8], voltage=vol[:8], params=params)

x_new = np.linspace(-2, 1, 1000)
#x_new[666] += 1e-4
params_new_pos = result_pos.params.values()
params_new_neg = result_neg.params.values()
cur_pred_continuous_pos = diode_model(x_new_pos := x_new[667:], *params_new_pos)
cur_pred_continuous_neg = diode_model(x_new_neg := x_new[:667], *params_new_neg)
print(cur_pred_continuous_pos[0])
fig, (axIV, axTrans) = plt.subplots(1, 2)
fig.set_size_inches(16, 9)
axIV.scatter(vol, np.abs(cur), label="Measured")
axIV.plot(x_new_pos, np.abs(np.abs(cur_pred_continuous_pos)), label='Fitted',
         color='red')
axIV.plot(x_new_neg, np.abs(np.abs(cur_pred_continuous_neg)), label='Fitted',
         color='red')
axIV.set_title("I-V Measured data + Fitted data")
axIV.set_yscale('log')
axIV.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
axIV.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.0, 0.2, 0.4, 0.6, 0.8), numticks=12))
axIV.yaxis.set_minor_formatter(ticker.NullFormatter())
axIV.tick_params(axis='both', direction='in', which='both')
axIV.set_xlabel('Voltage [V]')
axIV.set_ylabel('Log Absolute Current [A]')
axIV.legend(['Measured', f'Fitted'])
# -2V, -1V, 1V 지점에 전류값 표시
#for point in [-2, -1, 1]:
#    index = np.where(vol == point)[0][0]
#    x = vol[index]
#    y = np.abs(cur[index]) * 1.3
#    axIV.text(x, y, f"{y / 1.3:.4e}", ha="center", va="bottom", fontsize="large")

WavelengthSweep = soup.find_all("WavelengthSweep")
l_list = []
il_list = []
vol_list = []
for ws in WavelengthSweep:
    vol_list.append(ws.attrs['DCBias'] + "V")
    l_list.append([float(_l) for _l in ws.find("L").text.split(",")])
    il_list.append([float(_il) for _il in ws.find("IL").text.split(",")])
l = np.array(l_list)
il = np.array(il_list)
for i in range(len(l) - 1):
    axTrans.plot(l[i], il[i], label=vol_list[i])
axTrans.plot(l[-1], il[-1])
axTrans.set_title("Transmission spectra")
axTrans.legend(ncol=3, loc='lower center')
axTrans.set_xlabel('Wavelength [nm]')
axTrans.set_ylabel('Measured transmission [dB]')
axTrans.tick_params(axis='both', direction='in')
plt.show()
