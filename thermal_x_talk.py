import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def mw_to_dbm(mW):
    """This function converts a power given in mW to a power given in dBm."""
    return 10. * np.log10(mW)


def dbm_to_mw(dBm):
    """This function converts a power given in dBm to a power given in mW."""
    return 10 ** ((np.array(dBm)) / 10.)


def find_center(x, y):
    x, y = np.array(x), np.array(y)
    return np.sum(x * y) / np.sum(y)


def find_peak(x, y):
    x, y = np.array(x), np.array(y)
    index = np.argmax(y)
    return x[index]


class single_test:
    def __init__(self, SN, data):
        self.SN = SN
        self.data = data
        self.LD, self.Text, self.Tint, self.df, self.wavelength = self.pre_process()
        self.df_ta = self.time_avg()
        self.df_lr, self.x_talk = self.cal_x_talk()

    def pre_process(self):
        # get info
        df = pd.DataFrame(self.data)
        df = df[0].str.split('\t', expand=True).replace('\n', '', regex=True)  # 選取第0 column 並split with '\t'
        LD = df[0][df[0].str.match('Testing diode')].values[0][14]
        Text, Tint = df[0][df[0].str.match('Text')].values[0].split(', ')
        Text = float(Text.split(' = ')[-1])
        Tint = float(Tint.split(' = ')[-1])

        # clean data
        df.columns = df.iloc[df.index[df[0] == 'Wavelength'].tolist()[0]]  # 將有'Wavelength' 的那個row 設為 column name
        df = df.dropna(thresh=5)
        df = df[df['Wavelength'].apply(lambda x: x[0].isdigit())]  # 只保留 data['Wavelength'] 開頭是數字 的 rows
        df = df.dropna(axis='columns')
        df = df.astype('float')

        # get wavelength and re-index column names
        wavelength = df['Wavelength']
        cols = np.array([_.split('-') for _ in df.columns.values[1:]])
        cols_r, cols_b, cols_t = cols[:, 0], cols[:, 1], cols[:, 2]
        cols_new = pd.MultiIndex.from_arrays([cols_b, cols_r, cols_t])
        df2 = df.iloc[:, 1:]
        df2.columns = cols_new
        return LD, Text, Tint, df2.reset_index(drop=True), wavelength.reset_index(drop=True)

    def plot_spectrum(self):
        df = self.df.mean(axis=1, level=(0, 1))  # 將level 2 做平均 (平均times = 0~9)

        matplotlib.use('Agg')
        fig, ax = plt.subplots()
        for i in range(df.shape[1] - 1):
            ax.plot(self.wavelength, df.iloc[:, i + 1])

        ax.set_title('%s LD%s, Text: %.0f(C), Tint: %.0f(C)' % (self.SN, self.LD, self.Text, self.Tint))
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Spectrum (dBm)')
        ax.grid()

        fig.savefig(path + '%s LD%s spectrum, Text_%.0f, Tint_%.0f.png' % (self.SN, self.LD, self.Text, self.Tint))
        plt.close('all')

    def plot_center_wavelength(self):
        df_lr = self.df_lr
        df_ta = self.df_ta
        fig, ax = plt.subplots()
        for bias in df_ta['Bias(mA)'].unique():
            df_temp = df_ta.loc[df_ta['Bias(mA)'] == bias]
            x = np.linspace(df_temp['Ramp(mA)'].min(), df_temp['Ramp(mA)'].max(), 5)
            a = df_lr.loc[df_lr['Bias(mA)'] == bias, 'slope'].values[0]
            b = df_lr.loc[df_lr['Bias(mA)'] == bias, 'intercept'].values[0]
            y = a * x + b

            ax.scatter(df_temp['Ramp(mA)'], df_temp['Center_Wavelength(nm)'], label='%s mA' % int(bias))
            ax.plot(x, y, linestyle='--')

        ax.set_xlabel('Ramp Current (mA)')
        ax.set_ylabel('Center Wavelength (nm)')
        ax.set_title('%s LD%s, Text: %.0f(C), Tint: %.0f(C), x-talk: %.2f(C/A)'
                     % (self.SN, self.LD, self.Text, self.Tint, self.x_talk))
        ax.grid()
        ax.legend(loc='lower right')
        fig.savefig(path + '%s LD%s x-talk, Text_%.0f, Tint_%.0f.png' % (self.SN, self.LD, self.Text, self.Tint))
        plt.close('all')

    def time_avg(self):
        df = self.df.mean(axis=1, level=(0, 1))  # 將level 2 做平均 (平均times = 0~9)

        df2 = pd.DataFrame()
        for i, col in enumerate(df.columns):
            df2 = df2.append(pd.DataFrame({
                'Serial Number': self.SN, 'LD': self.LD, 'Text': self.Text, 'Tint': self.Tint,
                'Bias(mA)': float([i for i in col if i.startswith('B')][0][1:]),
                'Ramp(mA)': float([i for i in col if i.startswith('R')][0][1:]),
                'Center_Wavelength(nm)': find_center(self.wavelength, dbm_to_mw(df[col])),
                'Peak_Wavelength(nm)': find_peak(self.wavelength, dbm_to_mw(df[col]))
            }, index=[i]), ignore_index=True)

        return df2

    def cal_x_talk(self):
        df = self.df_ta

        df2 = pd.DataFrame()
        for i, bias in enumerate(df['Bias(mA)'].unique()):
            df_temp = df.loc[df['Bias(mA)'] == bias]
            x, y = df_temp['Ramp(mA)'], df_temp['Center_Wavelength(nm)']
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            df2 = df2.append(pd.DataFrame({
                'Serial Number': self.SN, 'LD': self.LD, 'Text': self.Text, 'Tint': self.Tint, 'Bias(mA)': bias,
                'slope': slope, 'intercept': intercept, 'r_value': r_value, 'p_value': p_value, 'std_err': std_err
            }, index=[i]), ignore_index=True)

        # slope (nm/A); factor: 0.34 (nm/C); x-talk (C/A)
        x_talk = (df2['slope'].max() - df2['slope'].min()) * 1000 / 0.34
        return df2, x_talk


class single_laser:
    def __init__(self, path, file):
        self.path = path
        self.file = file
        self.data, self.df_info, self.SN = self.read_file()
        self.df_ta, self.df_lr, self.df_xt = self.split_file()

    def read_file(self):
        path = self.path
        file = self.file
        SN = file[5:11]

        with open(path + file) as f:
            data = f.readlines()
        df_info = pd.DataFrame(data[:17])[0].str.split(' = ', expand=True).replace('\n', '', regex=True)

        return data, df_info, SN

    def split_file(self):
        data = self.data
        split_idx = [i for i, _ in enumerate(data) if _.startswith('Testing diode')]

        df_ta, df_lr, df_xt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in reversed(range(len(split_idx))):
            data_single = data[split_idx[i]:]
            data = data[:split_idx[i]]
            st = single_test(self.SN, data_single)
            print('%s LD%s, Text: %.0f(C), Tint: %.0f(C)' % (st.SN, st.LD, st.Text, st.Tint))

            df_ta = df_ta.append(st.df_ta)
            df_lr = df_lr.append(st.df_lr)

            df_xt = df_xt.append(pd.DataFrame({
                'Serial Number': st.SN, 'LD': st.LD, 'Text': st.Text, 'Tint': st.Tint, 'x_talk': st.x_talk
            }, index=[i]), ignore_index=True)

        return df_ta, df_lr, df_xt

    def plot_spectrum(self):
        data = self.data
        split_idx = [i for i, _ in enumerate(data) if _.startswith('Testing diode')]

        for i in reversed(range(len(split_idx))):
            print('%s / %s' % (len(split_idx) - i, len(split_idx)))
            data_single = data[split_idx[i]:]
            data = data[:split_idx[i]]
            st = single_test(self.SN, data_single)
            st.plot_spectrum()

    def plot_center_wavelength(self):
        data = self.data
        split_idx = [i for i, _ in enumerate(data) if _.startswith('Testing diode')]

        for i in reversed(range(len(split_idx))):
            print('%s / %s' % (len(split_idx) - i, len(split_idx)))
            data_single = data[split_idx[i]:]
            data = data[:split_idx[i]]
            st = single_test(self.SN, data_single)
            st.plot_center_wavelength()


if __name__ == '__main__':
    # path = '//li.lumentuminc.net/data/MIL/SJMFGING/Optics/Planar/Derek/D2/Thermal x-talk/New program/'
    path = 'C:/Users/cha75794/Desktop/x-talk/'

    files_ls = os.listdir(path)
    files_ls = [_ for _ in files_ls if _.endswith('.txt')]

    df_lr, df_ta, df_xt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for file in files_ls:
        data = single_laser(path, file)
        df_lr = df_lr.append(data.df_lr)
        df_ta = df_ta.append(data.df_ta)
        df_xt = df_xt.append(data.df_xt)
        # data.plot_center_wavelength()
        # data.plot_spectrum()

    df_lr.to_csv(path + 'Summary - linear regression.csv', index=False, header=True)
    df_ta.to_csv(path + 'Summary - time average.csv', index=False, header=True)
    df_xt.to_csv(path + 'Summary - center wavelength.csv', index=False, header=True)

    sns.stripplot(x='Text', y='x_talk', data=df_xt, jitter=True, hue='Serial Number', palette='Set1')
    plt.axhline(y=1, linewidth=1, ls='--', color='r')
    plt.axhline(y=-1, linewidth=1, ls='--', color='r')
    plt.xlabel('External Temperature (C)')
    plt.ylabel('Cross-Talk (C/A)')
    plt.legend(fontsize='small', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.title('Cross Talk')
    plt.tight_layout()
    plt.grid()
    plt.savefig(path + 'Summary - x-talk.png')
    plt.close()
