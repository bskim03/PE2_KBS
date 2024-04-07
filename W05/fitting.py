from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt

with open("E:\Dev\PE2\HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")

# IVMeasurement 추출
IVMeasurement = soup.find("IVMeasurement")

# voltage, current 저장
vol = np.array([float(v) for v in IVMeasurement.find("Voltage").text.split(",")])
cur = np.array([float(c) for c in IVMeasurement.find("Current").text.split(",")])

# 선형 회귀 모델 생성
model = np.poly1d(fit := np.polyfit(vol, cur, 12))

# 피팅 전류 값 생성
predicted_cur1 = model(vol)

# 연속 함수 만들기
x_space = np.linspace(-2, 1, 260)
predicted_cur2 = model(x_space)

# R 제곱 값 계산
mean_cur = np.mean(cur)
rss = np.sum((predicted_cur1 - mean_cur) ** 2)
tss = np.sum((cur - mean_cur) ** 2)
r_squared = rss / tss

# print("R^2 =", r_squared)

plt.figure(figsize=(12, 6))

# 측정 데이터 표시
plt.scatter(vol, np.abs(cur))

# 피팅 데이터 표시
plt.plot(vol, np.abs(predicted_cur1), label=f"R$^2$ = {r_squared}", color="red")

# -2V, -1V, 1V 지점에 전류값 표시
for point in [-2, -1, 1]:
    plt.text(vol[i := np.where(vol == point)[0][0]], y := np.abs(cur[i]), f"{y:.4e}", ha="center", va="bottom")

plt.legend(loc="center left")
plt.yticks([1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
plt.title("I-V Raw data + Fitted data")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
plt.yscale("log")

plt.show()