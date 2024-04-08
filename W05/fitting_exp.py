from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

with open("../HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")

# IVMeasurement 추출
IVMeasurement = soup.find("IVMeasurement")

# voltage, current 저장
vol = np.array([float(v) for v in IVMeasurement.find("Voltage").text.split(",")])
cur = np.array([float(c) for c in IVMeasurement.find("Current").text.split(",")])


# 다이오드 모델 정의
def diode_model(x, isat):
    return isat * (np.exp(x / 26e-3/2) - 1)


# 모델 계수 생성
vol2 = vol
vol2[8] = vol[8] + 1e-3
isat_forward = curve_fit(diode_model, vol2[8:], cur[8:])[0][0]
isat_backward = curve_fit(diode_model, vol2[:9], cur[:9])[0][0]

predicted_cur_forward = diode_model(vol[8:], isat_forward)
predicted_cur_backward = diode_model(vol[:9], isat_backward)


# R 제곱 값 계산
mean_cur = np.mean(cur)
rss = np.sum((predicted_cur_forward - mean_cur) ** 2)
tss = np.sum((cur - mean_cur) ** 2)
r_squared = rss / tss

plt.figure(figsize=(14, 6))

# 측정 데이터 표시
plt.scatter(vol, np.abs(cur))

# 피팅 데이터 표시
plt.plot(vol2[8:], np.abs(predicted_cur_forward), label=f"R$^2$ = {r_squared}", color="red")
plt.plot(vol2[:9], np.abs(predicted_cur_backward), color="red")
# plt.plot(x_space, np.abs(predicted_cur_continuous), color="red")

# -2V, -1V, 1V 지점에 전류값 표시
for point in [-2, -1, 1]:
    index = np.where(vol == point)[0][0]
    x = vol[index]
    y = np.abs(cur[index])
    plt.text(x, y, f"{y:.4e}", ha="center", va="bottom", fontsize="large")

plt.legend(loc="center left")
plt.title("I-V Raw data + Fitted data")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
plt.yscale("log")
plt.show()
