import matplotlib.pyplot as plt
import numpy as np
from .reporting import calculate_type1err_edd


def plot_type1err_edd(nncusum_results_dir,
                      onnc_results_dir,
                      onnr_results_dir,
                      hcusum_results_dir,
                      mewma_results_dir,
                      wlcusum_results_dir,
                      wlglr_results_dir,
                      type1err_list, f0_length):
    
    nncusum_stat_record = np.load(nncusum_results_dir)
    onnc_stat_record = np.load(onnc_results_dir)
    onnr_stat_record = np.load(onnr_results_dir)
    hcusum_stat_record = np.load(hcusum_results_dir)
    mewma_stat_record = np.load(mewma_results_dir)
    wlcusum_stat_record = np.load(wlcusum_results_dir)
    wlglr_stat_record = np.load(wlglr_results_dir)

    nncusum_type1err_edd_record = calculate_type1err_edd(nncusum_stat_record, type1err_list, f0_length)
    onnc_type1err_edd_record = calculate_type1err_edd(onnc_stat_record, type1err_list, f0_length)
    onnr_type1err_edd_record = calculate_type1err_edd(onnr_stat_record, type1err_list, f0_length)
    hcusum_type1err_edd_record = calculate_type1err_edd(hcusum_stat_record, type1err_list, f0_length)
    mewma_type1err_edd_record = calculate_type1err_edd(mewma_stat_record, type1err_list, f0_length)
    wlcusum_type1err_edd_record = calculate_type1err_edd(wlcusum_stat_record, type1err_list, f0_length)
    wlglr_type1err_edd_record = calculate_type1err_edd(wlglr_stat_record, type1err_list, f0_length)

    plt.rcParams['figure.figsize'] =  [3.26*2/2, 3]
    plt.rcParams.update({'font.size': 8})

    plt.plot(np.mean(nncusum_type1err_edd_record, axis=0), label="NN-CUSUM", c="red")
    plt.plot(np.mean(onnc_type1err_edd_record, axis=0), label="ONNC", c="orange", ls="dashed")
    plt.plot(np.mean(onnr_type1err_edd_record, axis=0), label="ONNR", c="purple", ls="dotted")
    plt.plot(np.mean(hcusum_type1err_edd_record, axis=0), label="Hotelling", c="grey", ls="dashdot")
    plt.plot(np.mean(mewma_type1err_edd_record, axis=0), label="MEWMA", c="blue", marker="o", markersize=4)
    plt.plot(np.mean(wlcusum_type1err_edd_record, axis=0), label="WL-CUSUM", c="skyblue", marker="^", markersize=4)
    plt.plot(np.mean(wlglr_type1err_edd_record, axis=0), label="WL-GLR", c="pink", marker="v", markersize=4)

    #plt.yticks(ticks=[1000,1500,2000,2500,3000], labels=[1000,1500,2000,2500,3000])
    #plt.xticks(ticks=np.array([0,1.5,4,6.5,9]), labels=["0.02","0.05", "0.10", "0.15" ,"0.20"])
    plt.legend()
    plt.xlabel("Type-I error")
    plt.ylabel("EDD")
    #plt.xlim(0,9)

    #plt.ylim(500,3100)
    plt.grid(ls="dotted")
    #plt.title("stride=10, window=100", fontsize=30)
    plt.legend(loc="lower left", fontsize=5.5)
    plt.savefig("C:/Users/jayao/Desktop/python_projects/ICASSP_NN_CPD/HASC_experiments/results/higgs_type1err_edd.pdf", bbox_inches = 'tight',pad_inches = 0.0)