import numpy as np
import numba
import yfinance as yf
import pandas as pd


@numba.jit(nopython=True, parallel=True)
def get_freq(v, window_size, sell_window_size, nbins=100):
    L = len(v)
    map_freq = [np.zeros(4 * nbins + 2) for i in range(4 * nbins + 2)]
    for i in numba.prange(max([0, L - window_size]), L):
        if i % 500 == 0:
            print(i)
        for j in range(i + 1, min([i + sell_window_size, L])):
            diff = v[j] - v[i]
            diff_pj = int(nbins * diff / v[i])
            for k in range(j + 1, min([i + sell_window_size, L])):
                diff_pk = int(nbins * (v[k] - v[j]) / v[j])
                map_freq[nbins + diff_pj][nbins + diff_pk] += 1
    return map_freq


@numba.jit(nopython=True, parallel=False)
def compute_expected_return(map_freq, target, ref_freq1, ref_freq2, nbins=100):
    expected_return = 0
    expected_loss = 0
    expected_acc = 0
    expected_loss_acc = 0

    freq_target = 0
    freq_below = 0
    for selected_freqi in range(len(map_freq)):
        selected_freq = selected_freqi - nbins
        if selected_freq < ref_freq1:
            continue
        if selected_freq > ref_freq2:
            continue
        for ki in range(len(map_freq[nbins + selected_freq])):
            k = ki - nbins
            prob = map_freq[nbins + selected_freq][k + nbins]
            if k >= target:
                freq_target += prob
                expected_return += k * prob
                expected_acc += prob
            else:
                freq_below += prob
                expected_loss += k * prob
                expected_loss_acc += prob
    expected_return /= expected_acc
    expected_loss /= expected_loss_acc
    freq_total = freq_target + freq_below
    prob_target = 100 * freq_target / freq_total
    return prob_target, expected_loss * 100 / nbins, expected_return * 100 / nbins


@numba.jit(nopython=True, parallel=False)
def bitcoin_agent(prices, wc, wv, epsc, epsv, USDTo, BTCo):
    USDT = USDTo
    BTC = BTCo
    Tc = 0
    Tv = 0
    N = len(prices)
    mean_pricec = np.mean(prices[0:wc])
    mean_pricev = np.mean(prices[0:wv])
    Pc = (1 - epsc) * mean_pricec
    Pv = (1 + epsv) * mean_pricev
    for t_global in range(max(wc, wv), N):
        current_price = prices[t_global]
        mean_pricec = np.mean(prices[t_global - wc : t_global])
        mean_pricev = np.mean(prices[t_global - wv : t_global])
        if t_global >= Tc + wc:
            Pc = (1 - epsc) * mean_pricec
        if t_global >= Tv + wv:
            Pv = (1 + epsv) * mean_pricev

        if USDT > 0:
            if current_price <= Pc:
                BTC += USDT / current_price
                USDT = 0
                Tc = t_global
        if BTC > 0:
            if current_price >= Pv:
                USDT += BTC * current_price
                BTC = 0
                Tv = t_global
    return [max([BTC * current_price, USDT]), BTC, USDT]


@numba.jit(nopython=True, parallel=True)
def optimize_agent(
    prices,
    USDTo,
    BTCo,
    range_wc=[int(1), int(500)],
    range_wv=[int(1), int(500)],
    range_epsc=[float(0), float(0.1)],
    range_epsv=[float(0.0), float(0.3)],
    n_scenarios=100,
):
    max_wallet_value = -1e30
    max_wc = 0
    max_wv = 0
    max_epsc = 0.0
    max_epsv = 0.0

    scenarios = []
    for i in range(n_scenarios):
        wc = np.random.randint(range_wc[0], range_wc[1])
        wv = np.random.randint(range_wv[0], range_wv[1])
        epsc = np.random.uniform(range_epsc[0], range_epsc[1])
        epsv = np.random.uniform(range_epsv[0], range_epsv[1])
        scenarios.append([wc, wv, epsc, epsv])

    results = np.zeros(n_scenarios, dtype=float)

    for i in numba.prange(n_scenarios):
        wc, wv, epsc, epsv = scenarios[i]
        wallet_value, BTC, USDT = bitcoin_agent(
            prices=prices, wc=wc, wv=wv, epsc=epsc, epsv=epsv, USDTo=USDTo, BTCo=BTCo
        )
        results[i] = wallet_value

    for i, scenario in enumerate(scenarios):
        wc, wv, epsc, epsv = scenario
        wallet_value = results[i]
        if wallet_value > max_wallet_value:
            max_wallet_value = wallet_value
            max_wc = wc
            max_wv = wv
            max_epsc = epsc
            max_epsv = epsv
            print(
                "wallet_value: ",
                max_wallet_value,
                " wc: ",
                max_wc,
                " wv: ",
                max_wv,
                " epsc: ",
                max_epsc,
                " epsv: ",
                max_epsv,
            )
    return [max_wallet_value, max_wc, max_wv, max_epsc, max_epsv]


def get_target_prices(prices, wc, wv, epsc, epsv):
    mean_pricec = np.mean(prices[len(prices) - int(wc) : len(prices)])
    mean_pricev = np.mean(prices[len(prices) - int(wv) : len(prices)])
    buy_price = mean_pricec * (1 - epsc)
    sell_price = mean_pricev * (1 + epsv)
    return buy_price, sell_price
