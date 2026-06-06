# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:18:03 2026

@author: Administrator

Created on Mon Apr 20 15:30:49 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
test_beta_necessity_final_P2_opt.py
功能：100% 宇宙学无关的 P2 物理模型验证 (gamma = gamma_0 + gamma_z * z)
包含：致命公式 Bug 修复 + 严格非物理值拦截 + 真正的 MLE 数值优化 + 角图绘制。
"""

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.special import gamma as gamma_func
import scipy.optimize as op
import emcee
import sys
import os
import time
from multiprocessing import Pool

# 导入用户自定义的绘图工具
try:
    import cosmo_tools
    HAS_COSMO_TOOLS = True
except ImportError:
    HAS_COSMO_TOOLS = False
    print("⚠️ 未检测到 cosmo_tools.py，将跳过绘图。")

# =============================================================================
# 1. 全局配置
# =============================================================================
C_LIGHT_KMS = 299792.458
RAD_PER_ARCSEC = np.pi / (180.0 * 3600.0)

N_WALKERS = 32
N_STEPS = 15000   # 保证充分收敛
BURN_IN = 5000   
USE_MULTIPROCESSING = True
SIGMA_SYS_FRAC = 0.03 

# =============================================================================
# 2. 数据加载 (100% 宇宙学无关，无需计算距离)
# =============================================================================
def load_data():
    csv_file = "matched_result_sgls.csv"
    if not os.path.exists(csv_file):
        print(f"❌ 错误: 找不到 {csv_file}")
        sys.exit(1)
        
    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]
    N = len(df)
    
    data_sgl = {
        'z_l': df['zl'].values,
        'z_s': df['zs'].values,
        'theta_E': df['thetaE'].values * RAD_PER_ARCSEC,
        'theta_ap': df['thetaap'].values * RAD_PER_ARCSEC,
        'sigma_ap': df['sigma_ap'].values,
        'dsigma_ap': df['dsigma_ap'].values,
        'delta': df['delta'].values  # 引入逐透镜光度斜率
    }

    data_sn = {
        'mb_l': df['sn_lens_m_b_corr'].values,
        'mb_s': df['sn_source_m_b_corr'].values,
    }

    cov_components = {}
    files = {'lens': 'cov_sgls_lens.txt', 'source': 'cov_sgls_source.txt', 'cross': 'cov_sgls_cross.txt'}
    for key, fname in files.items():
        if os.path.exists(fname):
            cov_components[key] = np.loadtxt(fname)
            
    return data_sgl, data_sn, cov_components, N

def get_combined_covariance(cov_comps):
    return cov_comps['lens'] + cov_comps['source'] - cov_comps['cross'] - cov_comps['cross'].T

# =============================================================================
# 3. 物理模型 (P2 框架) - 已修复所有数学 Bug
# =============================================================================
def predict_sigma(data_sgl, gamma_0, gamma_z, beta_ani, R_obs):
    gamma_val = gamma_0 + gamma_z * data_sgl['z_l']
    delta = data_sgl['delta']
    
    # 严格解耦定义
    xi = gamma_val + delta - 2.0
    
    try:
        t1 = gamma_func((xi - 1)/2.0) / gamma_func(xi/2.0)
        t2 = beta_ani * gamma_func((xi + 1)/2.0) / gamma_func((xi + 2)/2.0)
        t3 = (gamma_func(gamma_val/2.0) * gamma_func(delta/2.0)) / \
             (gamma_func((gamma_val-1)/2.0) * gamma_func((delta-1)/2.0))
             
        # 分母修正为 (3 - xi)
        denom = (xi - 2*beta_ani) * (3 - xi)
        
        if np.any(denom == 0) or np.any(~np.isfinite(t1)) or np.any(~np.isfinite(t2)) or np.any(~np.isfinite(t3)):
            raise ValueError
            
        F = ((3 - delta) / denom) * (t1 - t2) * t3
    except ValueError:
        return np.full_like(R_obs, np.inf)
    
    term_dist = 1.0 / R_obs
    term_dist[R_obs <= 0.001] = 0.0
    pre = (C_LIGHT_KMS**2) / (2 * np.sqrt(np.pi))
    
    sigma_sq = pre * term_dist * data_sgl['theta_E'] * F * \
               (data_sgl['theta_ap'] / data_sgl['theta_E'])**(2 - gamma_val)
               
    # 严格的非物理值拦截
    if np.any(~np.isfinite(sigma_sq)) or np.any(sigma_sq <= 0):
        return np.full_like(R_obs, np.inf)
        
    return np.sqrt(sigma_sq)

# =============================================================================
# 4. 纯似然函数 (MLE) 与 贝叶斯后验概率 (MAP)
# =============================================================================
def ln_likelihood(theta, data_sgl, data_sn, C_diff, fixed_beta=None):
    if fixed_beta is None:
        g0, gz, beta, f_int = theta
        if not (-1.0 < beta < 1.0): return -np.inf 
    else:
        g0, gz, f_int = theta
        beta = fixed_beta
        
    if not (1.0 < g0 < 3.0): return -np.inf
    if not (-2.0 < gz < 2.0): return -np.inf 
    if not (0.0 < f_int < 1.0): return -np.inf 
    
    gamma_val_check = g0 + gz * data_sgl['z_l']
    if np.any(gamma_val_check <= 1.05) or np.any(gamma_val_check >= 2.95):
        return -np.inf
    
    delta_mu = data_sn['mb_l'] - data_sn['mb_s']
    dist_ratio = 10.0**(0.2 * delta_mu)
    R_model = 1.0 - dist_ratio * ((1 + data_sgl['z_s']) / (1 + data_sgl['z_l']))
    
    if np.any(R_model <= 0.001): return -np.inf
    
    deriv = -0.2 * np.log(10.0) * (1.0 - R_model)
    Cov_R = deriv[:, None] * deriv[None, :] * C_diff
    
    sigma_pred = predict_sigma(data_sgl, g0, gz, beta, R_model)
    if np.any(~np.isfinite(sigma_pred)): return -np.inf
    
    res = data_sgl['sigma_ap'] - sigma_pred
    jac = -0.5 * sigma_pred / R_model
    Cov_sig_model = jac[:, None] * jac[None, :] * Cov_R
    
    diag_noise = data_sgl['dsigma_ap']**2 + (SIGMA_SYS_FRAC * data_sgl['sigma_ap'])**2 + (f_int * data_sgl['sigma_ap'])**2
    C_total = Cov_sig_model.copy()
    np.fill_diagonal(C_total, C_total.diagonal() + diag_noise)
    
    try:
        L = scipy.linalg.cho_factor(C_total, lower=True)
        sol = scipy.linalg.cho_solve(L, res)
        chi2 = np.dot(res, sol)
        log_det = 2 * np.sum(np.log(np.diag(L[0])))
        return -0.5 * (chi2 + log_det)
    except scipy.linalg.LinAlgError:
        return -np.inf

def ln_prob(theta, sgl, sn, cov, fixed_beta, beta_prior_type="flat"):
    lp = 0.0
    if fixed_beta is None and beta_prior_type == "gaussian":
        beta = theta[2] # P2 模型中，beta 是索引 2
        if not (-1.0 < beta < 1.0): return -np.inf
        lp += -0.5 * ((beta - 0.18) / 0.13)**2
        
    ll = ln_likelihood(theta, sgl, sn, cov, fixed_beta)
    if np.isinf(ll):
        return -np.inf
    return lp + ll

def neg_ln_likelihood(theta, data_sgl, data_sn, C_diff, fixed_beta):
    ll = ln_likelihood(theta, data_sgl, data_sn, C_diff, fixed_beta)
    return -ll if np.isfinite(ll) else 1e10

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    print("⏳ 正在加载 100% 宇宙学无关数据...")
    data_sgl, data_sn, cov_comps, N = load_data()
    C_diff = get_combined_covariance(cov_comps)
    print(f"✅ 成功加载 {N} 个 SGL-SN 配对系统。")
    
    scenarios = [
        {
            "name": "P2_Free_Beta_FlatPrior",
            "fixed_beta": None,
            "beta_prior_type": "flat",
            "names": ['gamma_0', 'gamma_z', 'beta_ani', 'f_int'],
            "labels": [r'\gamma_0', r'\gamma_z', r'\beta_{ani}', r'\delta_{int}'],
            "initial": [2.0, 0.0, -0.1, 0.1]
        },
        {
            "name": "P2_Free_Beta_GaussPrior",
            "fixed_beta": None,
            "beta_prior_type": "gaussian",
            "names": ['gamma_0', 'gamma_z', 'beta_ani', 'f_int'],
            "labels": [r'\gamma_0', r'\gamma_z', r'\beta_{ani}', r'\delta_{int}'],
            "initial": [2.0, 0.0, 0.18, 0.1]
        },
        {
            "name": "P2_Fixed_Beta_0",
            "fixed_beta": 0.0,
            "beta_prior_type": "none",
            "names": ['gamma_0', 'gamma_z', 'f_int'],
            "labels": [r'\gamma_0', r'\gamma_z', r'\delta_{int}'],
            "initial": [2.0, 0.0, 0.1]
        },
        {
            "name": "P2_Fixed_Beta_0_18",
            "fixed_beta": 0.18,
            "beta_prior_type": "none",
            "names": ['gamma_0', 'gamma_z', 'f_int'],
            "labels": [r'\gamma_0', r'\gamma_z', r'\delta_{int}'],
            "initial": [2.0, 0.0, 0.1]
        }
    ]
    
    results_summary = []
    comparison_samples = []

    print(f"\n🚀 开始最纯粹的 P2 动力学检验 (全量 Bug 修复版 + MLE 数值寻优)...")
    for config in scenarios:
        print(f"\n" + "="*50)
        print(f"🌟 正在运行模型: {config['name']}")
        
        initial = np.array(config['initial'])
        ndim = len(initial)
        pos = [initial + 1e-4 * np.random.randn(ndim) for i in range(N_WALKERS)]
        for p in pos: p[-1] = abs(p[-1]) 
            
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(N_WALKERS, ndim, ln_prob, 
                                            args=(data_sgl, data_sn, C_diff, config['fixed_beta'], config['beta_prior_type']),
                                            pool=pool if USE_MULTIPROCESSING else None)
            sampler.run_mcmc(pos, N_STEPS, progress=True)
            
        try:
            tau = sampler.get_autocorr_time(discard=BURN_IN, quiet=True)
            print(f"   📈 最大自相关时间 (tau): {np.max(tau):.1f} 步")
        except Exception as e:
            print("   ⚠️ 警告: 自相关时间计算失败，可能在边界附近未完全收敛。")
            
        samples = sampler.get_chain(discard=BURN_IN, thin=1, flat=True)
        
        param_constraints = {}
        for i, name in enumerate(config['names']):
            q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
            param_constraints[name] = f"{q50:.3f} (+{q84-q50:.3f}, -{q50-q16:.3f})"
            
        if config['fixed_beta'] is not None:
            param_constraints['beta_ani'] = f"{config['fixed_beta']:.3f} (fixed)"
            
        # 提取 3 个公共参数画对比图
        shared_samples = np.zeros((samples.shape[0], 3))
        if config['fixed_beta'] is None:
            shared_samples[:, 0] = samples[:, 0] # g0
            shared_samples[:, 1] = samples[:, 1] # gz
            shared_samples[:, 2] = samples[:, 3] # f_int (第4个)
        else:
            shared_samples[:, 0] = samples[:, 0] # g0
            shared_samples[:, 1] = samples[:, 1] # gz
            shared_samples[:, 2] = samples[:, 2] # f_int (第3个)
        comparison_samples.append(shared_samples)
        
        # MLE 寻优
        log_prob_samples = sampler.get_log_prob(discard=BURN_IN, thin=1, flat=True)
        mcmc_best_idx = np.argmax(log_prob_samples)
        mcmc_best_theta = samples[mcmc_best_idx]
        mcmc_lnL = ln_likelihood(mcmc_best_theta, data_sgl, data_sn, C_diff, config['fixed_beta'])
        
        print(f"   -> MCMC 寻找到的最大似然值: {mcmc_lnL:.2f}")
        print("   -> 正在执行数值优化 (Nelder-Mead)...")
        
        res_opt = op.minimize(neg_ln_likelihood, mcmc_best_theta, 
                              args=(data_sgl, data_sn, C_diff, config['fixed_beta']), 
                              method='Nelder-Mead')
        
        lnL_best = -res_opt.fun
        print(f"   ✅ 优化结束状态: {res_opt.success} ({res_opt.message})")
        print(f"   🎯 优化后的最终最大似然值: {lnL_best:.2f}")
        
        AIC = 2 * ndim - 2 * lnL_best
        BIC = ndim * np.log(N) - 2 * lnL_best
        
        results_summary.append({
            "Model": config['name'], "N_params": ndim, "Max ln(L)": lnL_best, 
            "AIC": AIC, "BIC": BIC, "params": param_constraints
        })

        if HAS_COSMO_TOOLS:
            try:
                stats_list = cosmo_tools.calculate_stats(samples, config['labels'])
                cosmo_tools.plot_getdist_advanced(samples, config['labels'], stats_list=stats_list, filename=f"{config['name']}_Opt_GetDist.pdf")
            except Exception as e:
                print(f"❌ 绘图失败: {e}")

    if HAS_COSMO_TOOLS:
        try:
            common_labels = [r'\gamma_0', r'\gamma_z', r'\delta_{int}']
            legend_names = ["P2 Free (Flat)", "P2 Free (Gauss)", "P2 Fixed 0", "P2 Fixed 0.18"]
            cosmo_tools.plot_getdist_comparison(comparison_samples, common_labels, legend_names, filename="P2_Beta_Models_Comparison_Opt.pdf")
        except Exception as e:
            print(f"❌ 对比图失败: {e}")

    print("\n\n" + "★" * 90)
    print("模型比较结果汇总 (P2 Framework, 100% Cosmo Independent, MLE Optimized)")
    print("★" * 90)
    print(f"{'Model Name':<28} | {'k':<5} | {'Max ln(L)':<12} | {'AIC':<10} | {'BIC':<10} | {'ΔBIC':<10}")
    print("-" * 95)
    
    min_bic = min([res['BIC'] for res in results_summary])
    for res in results_summary:
        delta_bic = res['BIC'] - min_bic
        print(f"{res['Model']:<28} | {res['N_params']:<5} | {res['Max ln(L)']:<12.2f} | {res['AIC']:<10.2f} | {res['BIC']:<10.2f} | {delta_bic:<10.2f}")
    print("-" * 95)

    print("\n\n" + "★" * 115)
    print("物理参数后验约束汇总 (Median and 68% Credible Intervals)")
    print("★" * 115)
    
    all_params = ['gamma_0', 'gamma_z', 'beta_ani', 'f_int']
    header = f"{'Model Name':<25} | " + " | ".join([f"{p:<18}" for p in all_params])
    print(header)
    print("-" * len(header))
    
    for res in results_summary:
        row = f"{res['Model']:<25} | "
        for p in all_params:
            if p in res['params']:
                row += f"{res['params'][p]:<18} | "
            else:
                row += f"{'N/A':<18} | "
        print(row[:-3])
    print("-" * len(header))
    print("\n✅ P2 终极修正版运行完毕！期待这份纯粹数据的表现！")