from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np

# XML 파일 불러오기
with open("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")

# IVMeasurement 추출
IVMeasurement = soup.find("IVMeasurement")

# voltage, current 저장
vol = [float(v) for v in IVMeasurement.find("Voltage").text.split(",")]
cur = [np.abs(float(c)) for c in IVMeasurement.find("Current").text.split(",")]

print("voltage:", vol)
print("current:", cur)

# plotting
plt.plot(vol, cur)
plt.xlabel("Voltage [V]")
plt.ylabel("Current [A]")
plt.yscale("log")
plt.title("I-V")
plt.grid(True)
plt.show()

# WavelengthSweep = soup.find_all("WavelengthSweep")
# for ws in WavelengthSweep:
#    l = list(map(float, ws.find("L").text.split(",")))
#    il = list(map(float, ws.find("IL").text.split(",")))
#    plt.plot(l, il)

# plt.show()
