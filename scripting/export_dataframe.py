import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime


def export(R2_ref, Max_ref, R2_IV, I):
    with open("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml", "r") as f:
        xml_data = f.read()

    soup = BeautifulSoup(xml_data, "xml")

    test_site_info = soup.find("TestSiteInfo").attrs # Batch, Column, Row, Masket, TestSite, Wafer 있음
    date = soup.find("PortCombo").attrs['DateStamp'].format('%Y %M %D')
    date_object = datetime.strptime(date, "%a %b %d %H:%M:%S %Y")
    formatted_date = date_object.strftime("%Y%m%d_%H%M%S")
    f.close()
    required_data = {
        'Batch': test_site_info['Batch'],
        'Wafer': test_site_info['Wafer'],
        'Mask': test_site_info['Maskset'],
        'TestSite': test_site_info['TestSite'],
        'Name': soup.find("Modulator").attrs['Name'],
        'Date': formatted_date,
        'Script ID': 'process LMZ',
        'Script Version': '0.1',
        'Script Owner': 'Group1_bskim',
        'Operator': soup.find("ModulatorSite").attrs['Operator'],
        'Row': test_site_info['DieRow'],
        'Column': test_site_info['DieColumn'],
        'ErrorFlag': '',
        'Error Description': '',
        'Analysis Wavelength': soup.find_all("DesignParameter")[1].text,
        'Rsq of Ref. spectrum (6th)': R2_ref,
        'Max transmission of Ref. spec. (dB)': Max_ref,
        'Rsq of IV': R2_IV,
        'I at -1V [A]': I[0],
        'I at 1V [A]': I[1],
    }
    df = pd.DataFrame(required_data, index=[0])
    df.to_csv("test_result.csv", index=False)
