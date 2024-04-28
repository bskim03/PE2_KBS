import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.ticker as ticker
import iv_process
import transmission_process
import export_dataframe

with open("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, "xml")
IVMeasurement = soup.find("IVMeasurement")
WavelengthSweep = soup.find_all("WavelengthSweep")

vol, cur = iv_process.get_data(IVMeasurement)
transmission, ref = transmission_process.get_data(WavelengthSweep)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))
# ax1: Measured transmission
# ax2: Fitted transmission reference
# ax3: Processed transmission reference
# ax4: Measured and fitted I-V characteristics
fontsize = 12

# --------------------------------- Measured transmission plotting --------------------------------- #
for i in range(len(transmission['vol'])):
    ax1.plot(transmission['l'][i], transmission['il'][i], label=transmission['vol'][i])
ax1.plot(ref['l'], ref['il'], label='reference')

ax1.legend(loc='lower center', ncol=4, fontsize=fontsize)
ax1.set_ylim(-52, -0)
ax1.set_title('Transmission spectra - As measured', fontsize=fontsize)
ax1.set_xlabel('Wavelength [nm]', fontsize=fontsize)
ax1.set_ylabel('Measured Transmission [dB]', fontsize=fontsize)
ax1.tick_params(axis='both', direction='in', labelsize=fontsize)

# --------------------------------- Fitted transmission reference plotting  --------------------------------- #
predicted_ils, ref_r2_scores, fit_labels = transmission_process.fit_ref(ref)

for i in range(len(predicted_ils)):
    ax2.plot(ref['l'], predicted_ils[i], label=fit_labels[i])
ax2.plot(ref['l'], ref['il'], label='Reference')

ax2.set_title('Transmission spectra reference - Fitted', fontsize=fontsize)
ax2.set_xlabel('Wavelength [nm]', fontsize=fontsize)
ax2.set_ylabel('Measured Transmission [dB]', fontsize=fontsize)
ax2.tick_params(axis='both', direction='in', labelsize=fontsize)
ax2.legend(loc="upper left", fontsize=fontsize, ncol=5)
ax2.set_ylim(-16.1, -5.9)

# --------------------------------- Processed transmission reference plotting  --------------------------------- #
processed_trans = transmission_process.process(transmission, ref)

for i in range(len(processed_trans['l'])):
    ax3.plot(processed_trans['l'][i], processed_trans['il'][i], label=processed_trans['vol'][i])
ax3.plot(ref['l'], ref['il'] - predicted_ils[3])

ax3.set_title('Flat Transmission spectra - As measured', fontsize=fontsize)
ax3.set_xlabel('Wavelength [nm]', fontsize=fontsize)
ax3.set_ylabel('Flat Measured Transmission [dB]', fontsize=fontsize)
ax3.tick_params(axis='both', direction='in', labelsize=fontsize)
ax3.legend()

# --------------------------------- IV characteristics plotting --------------------------------- #
x_new, cur_pred_continuous, r2_score_iv = iv_process.fit(vol, cur)

ax4.scatter(vol, np.abs(cur), label='Measured')
ax4.plot(x_new, np.abs(cur_pred_continuous), 'r', label=f'Fitted')

ax4.set_yscale('log')
ax4.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax4.yaxis.set_minor_locator(
    ticker.LogLocator(base=10, subs=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=10))
ax4.legend(fontsize=fontsize)
ax4.set_title('I-V Characteristics', fontsize=fontsize)
ax4.set_xlabel('Voltage [V]', fontsize=fontsize)
ax4.set_ylabel('Log Absolute Current [A]', fontsize=fontsize)
ax4.tick_params(axis='both', direction='in', which='both', labelsize=fontsize)

# 데이터프레임 내보내기
r2_ref_sixth = ref_r2_scores[5]
cur_minus1 = cur[np.where(vol == -1)[0][0]]
cur_plus1 = cur[np.where(vol == 1)[0][0]]

export_dataframe.export(r2_ref_sixth, max(ref['il']), r2_score_iv, (cur_minus1, cur_plus1))


plt.tight_layout()
plt.show()

# 할 일: 1. transmission 사인함수로 만들기
