from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt

with open("../HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")

WavelengthSweep = soup.find_all("WavelengthSweep")

for ws in WavelengthSweep[:-1]:
    vol: str = ws.attrs['DCBias'] + "V"
    l = np.array([float(_l) for _l in ws.find("L").text.split(",")])
    il = np.array([float(_il) for _il in ws.find("IL").text.split(",")])
    plt.plot(l, il, label=vol)

ws = WavelengthSweep[-1]
l = np.array([float(_l) for _l in ws.find("L").text.split(",")])
il = np.array([float(_il) for _il in ws.find("IL").text.split(",")])
plt.plot(l, il)

plt.title("Transmission spectra")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Measured transmission [dB]")
plt.legend(ncol=3, loc='lower center')
plt.show()
