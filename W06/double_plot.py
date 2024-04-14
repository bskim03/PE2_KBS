from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

with open("../HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")

IVMeasurement = soup.find("IVMeasurement")
vol = np.array([float(v) for v in IVMeasurement.find("Voltage").text.split(",")])
cur = np.array([float(c) for c in IVMeasurement.find("Current").text.split(",")])


def diode_model(v, isat, vt, n, p, q):
    return isat * (np.exp((v + p) / (vt * n)) - 1) + q
    # return isat * (np.exp(v / (vt * n)) - 1)


model = Model(diode_model)
params = model.make_params(isat=0, vt=26e-3, n=1, p=0, q=0)
params['n'].min = 1
result = model.fit(cur, v=vol, params=params)
print(result.fit_report())
print(result.params)

x_new = np.linspace(-2, 1, 1000)
#params_new = [-2.2811e-18, 0.03410632, 1, 0.7, 4.6117e-09]
params_new = result.params.values()
y_pred = diode_model(x_new, *params_new)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(vol, np.abs(cur))
plt.plot(x_new, np.abs(np.abs(y_pred)), label='Fitted data',
         color='red')
plt.title("I-V Measured data")
plt.yscale('log')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A]')
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')

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
plt.subplot(1, 2, 2)
for i in range(len(l) - 1):
    plt.plot(l[i], il[i], label=vol_list[i])
plt.plot(l[-1], il[-1])
plt.title("Transmission spectra")
plt.legend(ncol=3, loc='lower center')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.show()
