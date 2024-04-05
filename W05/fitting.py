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
model = np.poly1d(fit := np.polyfit(vol, cur, 6))

# 피팅 전류 값 생성
predicted_cur = model(vol)

# 연속 함수 만들기
x_space = np.linspace(-2, 1, 260)
predicted_cur2 = model(x_space)

# R 제곱 값 계산
mean_cur = np.mean(cur)
rss = np.sum((predicted_cur - mean_cur) ** 2)
tss = np.sum((cur - mean_cur) ** 2)
r_squared = rss / tss

# print("R^2 =", r_squared)

plt.title("I-V Characteristics")
# 측정 데이터 표시
plt.scatter(vol, np.abs(cur))

# 피팅 데이터 표시
plt.plot(x_space, np.abs(predicted_cur2), label=f"R$^2$ = {r_squared}", color="red")

plt.legend()
plt.yticks([1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
plt.yscale("log")
plt.show()
plt.plot()
