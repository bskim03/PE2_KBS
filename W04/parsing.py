from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

with open("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")

IVMeasurement = soup.find("IVMeasurement")
vol = list(map(float, IVMeasurement.find("Voltage").text.split(",")))
cur = list(map(float, IVMeasurement.find("Current").text.split(",")))
print("voltage:", vol)
print("current:", cur)


plt.plot(vol, cur)
plt.xlabel("Voltage [V]")
plt.ylabel("Current [A]")
plt.yscale("symlog")
plt.title("I-V")
plt.show()

# WavelengthSweep = soup.find_all("WavelengthSweep")
# for ws in WavelengthSweep:
#    l = list(map(float, ws.find("L").text.split(",")))
#    il = list(map(float, ws.find("IL").text.split(",")))
#    plt.plot(l, il)

# plt.show()
